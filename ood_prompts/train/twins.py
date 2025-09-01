import torch
from datasets import load_dataset

import os
import json
from tqdm import tqdm
import random
from omegaconf import OmegaConf

from evil_twins import load_model_tokenizer, DocDataset, optim_gcg

from ood_prompts.utils import split_data_among_gpus, get_logger
from ood_prompts.config import TwinsTrainConfig


logger = get_logger(__name__)


if __name__ == "__main__":
  config: TwinsTrainConfig = OmegaConf.merge(
    OmegaConf.structured(TwinsTrainConfig), OmegaConf.from_cli()
  )

  os.makedirs(config.output_path, exist_ok=True)

  logger.info(f"Launching training, full config:\n{OmegaConf.to_yaml(config)}")
  model, tokenizer = load_model_tokenizer(
    config.model_path,
    dtype=torch.bfloat16,  # torch.float32,
    use_flash_attn_2=False,
    device_map=f"cuda:{config.gpu_idx}",
  )

  if os.path.exists(config.prompts_dataset_path):
    logger.info(f"Loading existing prompts from {config.prompts_dataset_path}")
    with open(config.prompts_dataset_path, "r") as f:
      prompts = json.load(f)
  else:
    logger.info("Generating prompts dataset...")
    dataset = load_dataset(config.dataset_path)
    # pick random sequences from the dataset
    all_seqs = []
    for d in tqdm(dataset["test"], total=len(dataset["test"]), desc="Getting possible inputs..."):
      story = d["story"]
      story_words = story.split()

      if len(story_words) < config.min_sent_words:
        continue

      attempts = 0
      story_seqs = set()
      max_attempts = 100

      while len(story_seqs) < config.n_seqs_per_sample and attempts < max_attempts:
        seq_len = random.randint(config.min_sent_words, config.max_sent_words)
        if len(story_words) < seq_len:
          continue

        start_idx = random.randint(0, len(story_words) - seq_len)
        story_seq = " ".join(story_words[start_idx : start_idx + seq_len])
        story_seqs.add(story_seq)
        attempts += 1

      all_seqs.extend(list(story_seqs))

    logger.info("Building input/prompt dataset...")
    prompts = set()
    while len(prompts) < config.n_inputs and all_seqs:
      cur_seq = random.choice(all_seqs)
      prompts.add(cur_seq)
      all_seqs.remove(cur_seq)

    prompts = list(prompts)
    with open(f"{config.output_path}/prompts_dataset.json", "w") as f:
      json.dump(prompts, f, indent=2)

  # sort by longest first, to make sure we wont run out of mem
  prompts = sorted(prompts, key=lambda x: len(x), reverse=True)

  # if we've distributed the compute across many instances or we're resuming from a previous run
  if config.subset_start != -1:
    if config.subset_end == -1:
      prompts = prompts[config.subset_start :]
    else:
      prompts = prompts[config.subset_start : config.subset_end]

  optim_prompt = "x " * 20
  optim_prompt = optim_prompt.rstrip()
  data_for_current_gpu, start_idx = split_data_among_gpus(
    prompts,
    config.gpu_idx,
    config.n_gpus,
  )

  for i, prompt in enumerate(data_for_current_gpu):
    cur_idx = start_idx + i
    # if distributed since we're outputting to the same bucket, need to correctly track IDs
    if config.subset_start != -1:
      cur_idx = config.subset_start + i

    logger.info(f"Processing prompt {cur_idx}: {prompt}")
    dataset = DocDataset(
      model=model,
      tokenizer=tokenizer,
      orig_prompt=prompt,
      optim_prompt=optim_prompt,
      n_docs=config.n_docs,
      doc_len=config.doc_len,
      gen_batch_size=config.gen_batch_size,
      validate_prompt=False,
    )

    results, ids = optim_gcg(
      model=model,
      tokenizer=tokenizer,
      dataset=dataset,
      n_epochs=config.n_epochs,
      kl_every=1,
      log_fpath=f"{config.output_path}/twin_log_{cur_idx}.json",
      batch_size=config.batch_size,
      top_k=256,
      gamma=0.0,
      early_stop_kl=config.early_stop_kl,
    )

# %%
