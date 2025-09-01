from typing import Union
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from glob import glob
import json
import sys


def get_twin_results(
  base_dir: str,
  tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
  kl_thresh: float = sys.maxsize,
) -> dict:
  results_files = glob(f"{base_dir}/twin_log*.json")
  results = {}

  for fpath in results_files:
    cur_id = int(fpath.split("_")[-1].split(".")[0])
    try:
      cur_res = json.load(open(fpath, "r"))
    except json.JSONDecodeError:
      print("Failed to decode; skipping")
      continue

    orig_prompt = cur_res[0]["orig_prompt"]

    if cur_res[-1]["best_kl"] <= kl_thresh:
      best_kl = sys.maxsize
      best_prompt = ""
      for cur in cur_res:
        best_kl = min(best_kl, cur["best_kl"])
        if cur["best_kl"] == best_kl:
          best_prompt = cur["prompt"]

      orig_ids = tokenizer(orig_prompt, return_tensors="pt").input_ids
      twin_ids = torch.load(f"{base_dir}/twin_ids{cur_id}.pt", map_location="cpu")
      overlap_toks = set(orig_ids[0].tolist()) & set(twin_ids[0].tolist())

      # also add the BOS token if not in twin_ids b/c we made the new tokenizer like this
      if tokenizer.bos_token_id is not None:
        if twin_ids[0, 0] != tokenizer.bos_token_id:
          twin_ids = torch.cat(
            [torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long), twin_ids], dim=-1
          )

      results[cur_id] = {
        "best_kl": best_kl,
        "twin_str": best_prompt,
        "orig_str": orig_prompt,
        "twin_ids": twin_ids,
        "orig_ids": orig_ids,
        "overlap_toks": overlap_toks,
        "overlap_words": [tokenizer.decode([t]) for t in overlap_toks],
      }

  return results


def get_twin_results_new(base_dir: str, kl_thresh: float = sys.maxsize) -> dict:
  results_files = glob(f"{base_dir}/twin_log*.json")
  results = {}

  for fpath in results_files:
    cur_id = int(fpath.split("_")[-1].split(".")[0])
    try:
      cur_res = json.load(open(fpath, "r"))
    except json.JSONDecodeError:
      print(f"Failed to decode {fpath}; skipping")
      continue

    all_bests = [d["best_kl"] for d in cur_res]
    global_best = min(all_bests)

    if global_best <= kl_thresh:
      best_epoch = None
      for d in cur_res:
        if d["best_kl"] == global_best:
          best_epoch = d
          break
      if best_epoch:
        results[cur_id] = best_epoch

  return results


def get_suffix_results(
  base_dir: str,
  tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
  loss_thresh: float = sys.maxsize,
) -> dict:
  results_files = glob(f"{base_dir}/nanogcg_*.json")
  results = {}

  for fpath in results_files:
    cur_id = int(fpath.split("_")[-1].split(".")[0])
    cur_res = json.load(open(fpath, "r"))
    best_str = cur_res["best_string"]
    best_loss = cur_res["best_loss"]
    tgt = cur_res["target"]

    if best_loss <= loss_thresh:
      tgt_ids = tokenizer(tgt, return_tensors="pt").input_ids
      best_ids = tokenizer(best_str, return_tensors="pt").input_ids
      overlap_toks = set(tgt_ids[0].tolist()) & set(best_ids[0].tolist())

      cur_res["overlap_toks"] = [tokenizer.decode([t]) for t in overlap_toks]
      cur_res["target_ids"] = tgt_ids
      cur_res["best_ids"] = best_ids
      results[cur_id] = cur_res

  return results
