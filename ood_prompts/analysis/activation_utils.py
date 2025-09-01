import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  GPTNeoXForCausalLM,
  Gemma2ForCausalLM,
  LlamaForCausalLM,
)
import numpy as np
from numpy import ndarray
from jaxtyping import Float, jaxtyped, Int64
from beartype import beartype as typechecker
from nnsight import LanguageModel

import torch.nn as nn
from transformers import PretrainedConfig

# from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXSdpaAttention

# TODO: change back to older version of transformers
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention  # , Gemma2SdpaAttention
from transformers.models.llama.modeling_llama import LlamaAttention  # , LlamaSdpaAttention

from tqdm import tqdm
import random

DataOutput = list[list[Tensor]]


# this is no longer needed... we can simply use attn_implementation="eager"
# def replace_sdpa_attn(model: nn.Module, config: PretrainedConfig):
#   for name, child in model.named_children():
#     if isinstance(child, GPTNeoXSdpaAttention):
#       new_attention = GPTNeoXAttention(config)

#       new_attention.query_key_value.weight = child.query_key_value.weight
#       new_attention.query_key_value.bias = child.query_key_value.bias
#       new_attention.dense.weight = child.dense.weight
#       new_attention.dense.bias = child.dense.bias
#       setattr(model, name, new_attention)
#     elif isinstance(child, Gemma2Attention) or isinstance(child, LlamaAttention):
#       new_attention = (
#         Gemma2Attention(config) if isinstance(child, Gemma2Attention) else LlamaAttention(config)
#       )

#       new_attention.q_proj.weight = child.q_proj.weight
#       new_attention.q_proj.bias = child.q_proj.bias
#       new_attention.k_proj.weight = child.k_proj.weight
#       new_attention.k_proj.bias = child.k_proj.bias
#       new_attention.v_proj.weight = child.v_proj.weight
#       new_attention.v_proj.bias = child.v_proj.bias
#       new_attention.o_proj.weight = child.o_proj.weight
#       new_attention.o_proj.bias = child.o_proj.bias

#       setattr(model, name, new_attention)
#     else:
#       replace_sdpa_attn(child, config)


def load_nnsight_model(
  model_name: str, replace_attn: bool = True, device: str = "cuda:0"
) -> LanguageModel:
  # replace attn is deprecated
  # hf_model, tokenizer = load_model_tokenizer(model_name, torch.bfloat16, "cpu", eval_mode=True)
  hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, attn_implementation="eager" if replace_attn else None
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # if replace_attn and (
  #   isinstance(hf_model, GPTNeoXForCausalLM)
  #   or isinstance(hf_model, Gemma2ForCausalLM)
  #   or isinstance(hf_model, LlamaForCausalLM)
  # ):
  #   replace_sdpa_attn(hf_model, hf_model.config)
  # elif replace_attn:
  #   raise ValueError(f"Model type {type(hf_model)} not supported")

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  model = LanguageModel(
    hf_model,
    tokenizer=tokenizer,
    dispatch=True,
  ).to(device)
  print(model)
  print(model.device)
  print(model.dtype)

  return model


def compute_mean_diff(
  orig_examples: DataOutput,
  twin_examples: DataOutput,
  num_layers: int,
  hidden_size: int,
  device: str = "cuda:0",
) -> Float[ndarray, "num_layers hidden_size"]:
  res_scores = torch.zeros(num_layers, hidden_size, device=device)

  for i in tqdm(range(num_layers), total=num_layers, desc="Computing mean diffs (batched)"):
    cur_origs = orig_examples[i]
    cur_twins = twin_examples[i]

    P = len(cur_origs)
    N = len(cur_twins)
    if P == 0 or N == 0:
      continue

    orig_stack = torch.stack(cur_origs, dim=0).to(device)
    twin_stack = torch.stack(cur_twins, dim=0).to(device)

    mean_orig = orig_stack.mean(dim=0)
    mean_twin = twin_stack.mean(dim=0)

    layer_diff = mean_orig - mean_twin

    res_scores[i, :] = layer_diff

  return res_scores.cpu().numpy()


def prepare_data(
  data,
  model,
  num_layers: int,
  analysis_type: str = "overall",
  choose_rand_toks: bool = False,
) -> tuple[DataOutput, DataOutput]:
  """
  Collects the hidden states from each layer for the original and twin examples,
  based on the analysis_type.
  """
  orig_examples = [[] for _ in range(num_layers)]
  twin_examples = [[] for _ in range(num_layers)]

  for item in tqdm(data, total=len(data), desc=f"preparing data ({analysis_type})"):
    orig_output = item["mlp_out_orig"].to(model.device)
    twin_output = item["mlp_out_twin"].to(model.device)

    layer_id = item["layer"]

    if analysis_type == "overall":
      if choose_rand_toks:
        # pick random tokens
        rand_count = 6
        orig_indices = torch.randint(0, orig_output.shape[0], (rand_count,))
        twin_indices = torch.randint(0, twin_output.shape[0], (rand_count,))
      else:
        # last token
        orig_indices = [orig_output.shape[0] - 1]
        twin_indices = [twin_output.shape[0] - 1]
      for idx in orig_indices:
        orig_examples[layer_id].append(orig_output[idx])
      for idx in twin_indices:
        twin_examples[layer_id].append(twin_output[idx])

    elif analysis_type == "baseline":
      # random half of the time put original into orig_examples,
      # otherwise put it into twin_examples
      add_to_orig = random.choice([True, False])
      if choose_rand_toks:
        rand_count = 6
        orig_indices = torch.randint(0, orig_output.shape[0], (rand_count,))
      else:
        orig_indices = [orig_output.shape[0] - 1]

      if add_to_orig:
        for idx in orig_indices:
          orig_examples[layer_id].append(orig_output[idx])
      else:
        for idx in orig_indices:
          twin_examples[layer_id].append(orig_output[idx])

  return orig_examples, twin_examples


@jaxtyped(typechecker=typechecker)
def get_top_features_for_layers(
  mean_diffs: Float[ndarray, "num_layers hidden_size"], k: int, least_important: bool = False
) -> dict[int, Int64[ndarray, " k"]]:
  """
  Returns a dictionary mapping layer -> list of top k feature indices
  according to the magnitude of the mean differences.
  """
  num_layers = mean_diffs.shape[0]
  layer_to_top_feats = {}

  for layer_id in range(num_layers):
    if least_important:
      top_k = np.argsort(np.abs(mean_diffs[layer_id]))[:k]
    else:
      top_k = np.argsort(np.abs(mean_diffs[layer_id]))[-k:]

    layer_to_top_feats[layer_id] = top_k

  return layer_to_top_feats


@jaxtyped(typechecker=typechecker)
def get_random_features_for_layers(
  hidden_size: int, k: int, num_layers: int
) -> dict[int, Int64[ndarray, " k"]]:
  """
  Returns a dictionary mapping layer -> random list of k feature indices.
  """
  layer_to_rand_feats = {}
  for layer_id in range(num_layers):
    rand_feats = np.random.choice(hidden_size, size=k, replace=False)
    layer_to_rand_feats[layer_id] = rand_feats
  return layer_to_rand_feats


class PromptDataset(Dataset):
  def __init__(self, all_input_ids):
    self.all_input_ids = all_input_ids  # list of Tensors

  def __len__(self):
    return len(self.all_input_ids)

  def __getitem__(self, idx):
    x = self.all_input_ids[idx]
    # If x is shape (1, seq_len), convert to (seq_len,)
    if x.dim() == 2 and x.size(0) == 1:
      x = x.squeeze(0)
    return x


def collate_fn(batch, pad_id=0):
  """
  We ensure everything is placed on CPU inside the collate function.
  """
  # Possibly convert each example to CPU if it isn't already
  batch_cpu = []
  for x in batch:
    if x.device.type != "cpu":
      x = x.cpu()
    batch_cpu.append(x)

  # Now batch_cpu all on CPU
  max_len = max(t.size(0) for t in batch_cpu)
  padded_list = []
  lengths = []

  for x in batch_cpu:
    lengths.append(x.size(0))
    needed = max_len - x.size(0)
    if needed > 0:
      pad_tensor = torch.full((needed,), pad_id, dtype=x.dtype, device="cpu")  # on CPU
      x = torch.cat([x, pad_tensor], dim=0)  # now valid cat
    padded_list.append(x)

  padded_batch = torch.stack(padded_list, dim=0)  # shape: (batch, max_len) on CPU
  lengths = torch.tensor(lengths, dtype=torch.long, device="cpu")  # keep on CPU

  return padded_batch, lengths
