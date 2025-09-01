import torch
from torch import Tensor
from transformers import (
  AutoTokenizer,
  GPTNeoXForCausalLM,
  Gemma2ForCausalLM,
  LlamaForCausalLM,
  Qwen2ForCausalLM,
)
from nnsight import LanguageModel
from nnsight.envoy import Envoy

from jaxtyping import Float, jaxtyped, Int64
from beartype import beartype as typechecker
from einops import rearrange

import pickle
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from dataclasses import dataclass

from ood_prompts.analysis import (
  # get_twin_results,
  get_twin_results_new,
  get_suffix_results,
  load_nnsight_model,
)
from ood_prompts.config import ExtractHiddenConfig
from ood_prompts.utils import get_logger

from transformers.utils import logging

logging.set_verbosity(40)

logger = get_logger(__name__)


@jaxtyped(typechecker=typechecker)
@dataclass
class LayerOutputs:
  mlp_out_orig: Float[Tensor, "num_toks_orig hidden_size"]
  mlp_out_twin: Float[Tensor, "num_toks_twin hidden_size"]

  attn_orig: Float[Tensor, "num_heads num_toks_orig num_toks_orig"]
  attn_twin: Float[Tensor, "num_heads num_toks_twin num_toks_twin"]


@jaxtyped(typechecker=typechecker)
def extract_hidden_gptneox(
  orig: Int64[Tensor, "1 num_toks_orig"],
  twin: Int64[Tensor, "1 num_toks_twin"],
  model: LanguageModel,
  layers,
) -> list[LayerOutputs]:
  with model.trace(orig, output_attentions=True, use_cache=False):
    attn_orig = [layer.attention.output.save() for layer in layers]
    hidden_states_orig = [layer.mlp.output.save() for layer in layers]

  with model.trace(twin, output_attentions=True, use_cache=False):
    attn_twin = [layer.attention.output.save() for layer in layers]
    hidden_states_twin = [layer.mlp.output.save() for layer in layers]

  # [2] has the attn weights per head for gptneox
  return [
    LayerOutputs(
      mlp_out_orig=hidden_states_orig[i][0].detach().cpu().clone(),
      mlp_out_twin=hidden_states_twin[i][0].detach().cpu().clone(),
      attn_orig=attn_orig[i][2][0].detach().cpu().clone(),
      attn_twin=attn_twin[i][2][0].detach().cpu().clone(),
    )
    for i in range(len(layers))
  ]


def extract_hidden_gemma_llama_qwen(
  orig: Int64[Tensor, "1 num_toks_orig"],
  twin: Int64[Tensor, "1 num_toks_twin"],
  model: LanguageModel,
  layers: list[Envoy],
) -> list[LayerOutputs]:
  with model.trace(orig, output_attentions=True, use_cache=False):
    attn_orig = [layer.self_attn.output.save() for layer in layers]
    hidden_states_orig = [layer.mlp.output.save() for layer in layers]

  with model.trace(twin, output_attentions=True, use_cache=False):
    attn_twin = [layer.self_attn.output.save() for layer in layers]
    hidden_states_twin = [layer.mlp.output.save() for layer in layers]

  # [1] has the attn weights per head
  return [
    LayerOutputs(
      mlp_out_orig=hidden_states_orig[i][0].detach().cpu().clone(),
      mlp_out_twin=hidden_states_twin[i][0].detach().cpu().clone(),
      attn_orig=attn_orig[i][1][0].detach().cpu().clone(),
      attn_twin=attn_twin[i][1][0].detach().cpu().clone(),
    )
    for i in range(len(layers))
  ]


if __name__ == "__main__":
  config: ExtractHiddenConfig = OmegaConf.merge(
    OmegaConf.structured(ExtractHiddenConfig), OmegaConf.from_cli()
  )
  logger.info(f"Config: {config}")

  tokenizer = AutoTokenizer.from_pretrained(config.model_name)
  base_dir = os.path.join(config.base_optim_dir, "")

  if config.use_suffix:
    results = get_suffix_results(base_dir, tokenizer, config.loss_thresh)
  else:
    # results = get_twin_results(base_dir, tokenizer, config.kl_thresh)
    results = get_twin_results_new(base_dir, config.kl_thresh)

  model = load_nnsight_model(config.model_name, True, "cuda:0")
  hidden_states_by_input = []

  # move all to GPU first to optimize performance
  for k, v in results.items():
    results[k]["orig_ids"] = rearrange(
      torch.tensor(v["orig_prompt"]["ids"], dtype=torch.long, device=model.device), "n -> 1 n"
    )
    results[k]["twin_ids"] = rearrange(
      torch.tensor(v["optim_prompt"]["ids"], dtype=torch.long, device=model.device), "n -> 1 n"
    )
    # if "orig_ids" in v:
    #   v["orig_ids"] = v["orig_ids"].to(model.device)
    # elif "target_ids" in v:
    #   v["orig_ids"] = v["target_ids"].to(model.device)
    # else:
    #   raise KeyError(f"'orig_ids' or 'target_ids' missing for result {k}")

    # if "twin_ids" in v:
    #   v["twin_ids"] = v["twin_ids"].to(model.device)
    # elif "best_ids" in v:
    #   v["twin_ids"] = v["best_ids"].to(model.device)
    # else:
    #   raise KeyError(f"'twin_ids' or 'best_ids' missing for result {k}")

  # if random, replace the twin with random tokens
  if config.replace_rand:
    for k, v in results.items():
      v["twin_ids"] = torch.randint(
        0, model._model.config.vocab_size, v["twin_ids"].shape, device=model.device
      )

  for i, (k, v) in tqdm(
    enumerate(results.items()),
    total=len(results),
    desc="Getting hidden states",
  ):
    orig = v["orig_ids"]
    twin = v["twin_ids"]

    hf_model = model._model

    if isinstance(hf_model, GPTNeoXForCausalLM):
      layers = model.gpt_neox.layers
    elif (
      isinstance(hf_model, Gemma2ForCausalLM)
      or isinstance(hf_model, LlamaForCausalLM)
      or isinstance(hf_model, Qwen2ForCausalLM)
    ):
      layers = model.model.layers
    else:
      raise ValueError(f"Model type {type(hf_model)} not supported")

    if isinstance(hf_model, GPTNeoXForCausalLM):
      layer_outs = extract_hidden_gptneox(orig, twin, model, layers)
    elif (
      isinstance(hf_model, Gemma2ForCausalLM)
      or isinstance(hf_model, LlamaForCausalLM)
      or isinstance(hf_model, Qwen2ForCausalLM)
    ):
      layer_outs = extract_hidden_gemma_llama_qwen(orig, twin, model, layers)
    else:
      raise ValueError(f"Model type {type(hf_model)} not supported")

    for layer_idx, layer_out in enumerate(layer_outs):
      hidden_states_by_input.append(
        {
          "orig_ids": v["orig_ids"],
          "twin_ids": v["twin_ids"],
          "orig_str": v["orig_prompt"]["text"],
          "twin_str": v["optim_prompt"]["text"],
          "layer": layer_idx,
          "mlp_out_orig": layer_out.mlp_out_orig,
          "mlp_out_twin": layer_out.mlp_out_twin,
          "attn_orig": layer_out.attn_orig,
          "attn_twin": layer_out.attn_twin,
          "overlap_toks": v.get("overlap_toks", []),
        }
      )

  save_name = f"{config.output_dir}/hidden_states_by_input.pkl"
  if config.replace_rand:
    save_name = f"{config.output_dir}/hidden_states_by_input_RANDOM.pkl"

  pickle.dump(hidden_states_by_input, open(save_name, "wb"))
