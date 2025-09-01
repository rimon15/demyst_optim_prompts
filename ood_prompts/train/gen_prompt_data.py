from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from omegaconf import OmegaConf
import json
import os
from tqdm import tqdm
import random

import nltk

nltk.download("punkt", quiet=True)
from nltk import sent_tokenize

from ood_prompts.config import GenPromptDataConfig
from ood_prompts.utils import get_logger

logger = get_logger(__name__)


def process_dolly15k(
  data: Dataset,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  n_prompts: int,
  min_prompt_len: int,
  max_prompt_len: int,
) -> list[str]:
  logger.info("Processing Dolly15k prompts...")
  filtered_prompts = []
  for d in tqdm(data, total=len(data)):
    if (
      d["category"] == "brainstorming"
      and len(tokenizer.encode(d["instruction"])) <= max_prompt_len
      and len(tokenizer.encode(d["instruction"])) >= min_prompt_len
    ):
      filtered_prompts.append(d["instruction"])

    if len(filtered_prompts) >= n_prompts:
      break

  logger.info(f"Total filtered prompts: {len(filtered_prompts)}")
  return filtered_prompts


def process_openhermes(
  data: Dataset,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  n_prompts: int,
  min_prompt_len: int,
  max_prompt_len: int,
) -> list[str]:
  logger.info("Processing OpenHermes prompts...")
  filtered_prompts = []
  data = data["train"]

  for d in tqdm(data, total=len(data)):
    if d["category"] == "coding" or d["category"] == "general":
      prompt = d["conversations"][0]["value"]

      if (
        len(tokenizer.encode(prompt)) <= max_prompt_len
        and len(tokenizer.encode(prompt)) >= min_prompt_len
      ):
        filtered_prompts.append(prompt)

    if len(filtered_prompts) >= n_prompts:
      break

  logger.info(f"Total filtered prompts: {len(filtered_prompts)}")
  return filtered_prompts


def process_alpaca(
  data: Dataset,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  n_prompts: int,
  min_prompt_len: int,
  max_prompt_len: int,
) -> list[str]:
  logger.info("Processing Alpaca prompts...")
  filtered_prompts = []
  data = data["train"]

  for d in tqdm(data, total=len(data)):
    prompt = d["instruction"]

    if (
      d["input"] == ""
      and prompt.isascii()
      and len(tokenizer.encode(prompt)) <= max_prompt_len
      and len(tokenizer.encode(prompt)) >= min_prompt_len
    ):
      filtered_prompts.append(prompt)

    if len(filtered_prompts) >= n_prompts:
      break

  logger.info(f"Total filtered prompts: {len(filtered_prompts)}")
  return filtered_prompts


def process_wiki(
  data: Dataset,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  n_prompts: int,
  min_prompt_len: int,
  max_prompt_len: int,
) -> list[str]:
  logger.info("Processing Wiki/webtext prompts...")
  filtered_prompts = []
  data = data.shuffle()

  for d in tqdm(data, total=len(data)):
    text = d["text"]

    if (
      len(text.split()) < min_prompt_len
      or "= =" in text
      or "@-@" in text
      or "@,@" in text
      or "@.@" in text
      or "<unk>" in text
      or "= " in text
    ):
      continue

    sentences = nltk.sent_tokenize(text)
    random.shuffle(sentences)

    for sentence in sentences:
      if not sentence.isascii():
        continue

      words = sentence.split()
      if len(words) < 3:
        continue

      found_chunk = False
      for end_idx in range(2, len(words) + 1):
        chunk = " ".join(words[:end_idx])
        token_length = len(tokenizer.encode(chunk))
        if min_prompt_len <= token_length <= max_prompt_len:
          filtered_prompts.append(chunk)
          found_chunk = True
          break

      if found_chunk:
        break

    if len(filtered_prompts) >= n_prompts:
      break

  logger.info(f"Total filtered prompts: {len(filtered_prompts)}")
  return filtered_prompts


if __name__ == "__main__":
  config: GenPromptDataConfig = OmegaConf.merge(
    OmegaConf.structured(GenPromptDataConfig), OmegaConf.from_cli()
  )

  tokenizer = AutoTokenizer.from_pretrained(config.model_name)

  if config.dataset_name == "Salesforce/wikitext":
    dataset = load_dataset(config.dataset_name, "wikitext-103-v1", split="train")
  else:
    dataset = load_dataset(config.dataset_name)

  os.makedirs(os.path.dirname(config.out_path), exist_ok=True)
  if config.dataset_name == "databricks/databricks-dolly-15k":
    filtered_prompts = process_dolly15k(
      dataset["train"], tokenizer, config.n_prompts, config.min_prompt_len, config.max_prompt_len
    )
  elif config.dataset_name == "teknium/OpenHermes-2.5":
    filtered_prompts = process_openhermes(
      dataset, tokenizer, config.n_prompts, config.min_prompt_len, config.max_prompt_len
    )
  elif config.dataset_name == "tatsu-lab/alpaca":
    filtered_prompts = process_alpaca(
      dataset, tokenizer, config.n_prompts, config.min_prompt_len, config.max_prompt_len
    )
  elif config.dataset_name == "Salesforce/wikitext" or "openwebtext" in config.dataset_name:
    if "openwebtext" in config.dataset_name:
      dataset = dataset["train"]

    filtered_prompts = process_wiki(
      dataset, tokenizer, config.n_prompts, config.min_prompt_len, config.max_prompt_len
    )
  else:
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")

  with open(config.out_path, "w") as f:
    json.dump(filtered_prompts, f, indent=2)
