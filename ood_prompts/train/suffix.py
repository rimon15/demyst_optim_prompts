import os
from datasets import load_dataset
from argparse import ArgumentParser
import random
import json

import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from ood_prompts.utils import split_data_among_gpus


MODEL_PATH = os.getenv("MODEL_PATH")  # , "google/gemma-2-2b-it")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
MAX_TARGET_LEN = 8
MAX_PROMPT_LEN = 20
# SUFFIX_LEN = 20

model = AutoModelForCausalLM.from_pretrained(
  MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# instruct dataset
dataset = load_dataset("databricks/databricks-dolly-15k")
filtered_prompts = []
for d in dataset["train"]:
  if d["category"] == "brainstorming" and len(tokenizer.encode(d["instruction"])) <= MAX_PROMPT_LEN:
    # filtered_prompts.append((d["instruction"], d["response"]))
    filtered_prompts.append(d["instruction"])
print(f"tot filtered instruct prompts: {len(filtered_prompts)}")


# targets dataset
cola = load_dataset("nyu-mll/glue", "cola")
filtered_targets = []
for d in cola["train"]:
  if d["label"] == 1:
    filtered_targets.append(d["sentence"])
for d in cola["validation"]:
  if d["label"] == 1:
    filtered_targets.append(d["sentence"])
for d in cola["test"]:
  if d["label"] == 1:
    filtered_targets.append(d["sentence"])
cola_filtered = [x for x in filtered_targets if len(tokenizer.encode(x)) <= MAX_TARGET_LEN]
print(f"tot filtered targets: {len(cola_filtered)}")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--gpu_idx", type=int)
  parser.add_argument("--n_gpus", type=int)
  args = parser.parse_args()

  data_for_current_gpu, start_idx = split_data_among_gpus(
    filtered_prompts, args.gpu_idx, args.n_gpus
  )
  for i, prompt in enumerate(data_for_current_gpu):  # tqdm(, desc="Training nanogcg"):
    optim_tgt = random.choice(cola_filtered)
    print(f"cur idx: {i + start_idx}; prompt: {prompt}; target: {optim_tgt}")

    config = GCGConfig(
      num_steps=2000,
      # optim_str_init=torch.randint(0, tokenizer.vocab_size, ()),
      search_width=512,
      # batch_size=4,
      topk=256,
      n_replace=1,
      buffer_size=0,
      use_mellowmax=False,
      use_prefix_cache=True,
      allow_non_ascii=False,
      filter_ids=True,
      add_space_before_target=False,
      seed=42,
      verbosity="INFO",
      early_stop=True,
    )
    result = nanogcg.run(model, tokenizer, prompt, optim_tgt, config)

    with open(f"{OUTPUT_PATH}/nanogcg_log_{i + start_idx}.json", "w") as f:
      result_json = {
        "message": prompt,
        "target": optim_tgt,
        "best_loss": result.best_loss,
        "best_string": result.best_string,
        "losses": result.losses,
        "strings": result.strings,
      }
      json.dump(result_json, f, indent=2)
