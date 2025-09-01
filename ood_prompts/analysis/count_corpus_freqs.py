# Given a training dataset, count the frequency of each token in the dataset

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset

from collections import Counter
from tqdm import tqdm
import json
from omegaconf import OmegaConf
import multiprocessing as mp

from ood_prompts.config import CorpusCountConfig
from ood_prompts.utils import get_logger
from ood_prompts.ext.count_utils import count_tokens


logger = get_logger(__name__)


def process_batch(args) -> Counter:
  indices, dataset = args
  batch_toks = [dataset[i]["input_ids"] for i in indices]

  return count_tokens(batch_toks)


def merge_counts(counts_list: list[Counter]) -> Counter:
  final_counts = Counter()
  for counts in tqdm(counts_list, desc="Merging counts"):
    final_counts.update(counts)

  sorted_counts = Counter(dict(final_counts.most_common()))
  return sorted_counts


def get_stats(
  dataset_name: str,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  output_dir: str,
  batch_size: int,
  num_proc: int,
) -> None:
  train_data = load_dataset(dataset_name, split="train")
  logger.info("Processing dataset...")

  indices = list(range(len(train_data)))
  batch_ids = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

  with tqdm(total=len(batch_ids), desc="Counting tokens") as pbar:
    with mp.Pool(num_proc) as pool:
      results = []
      for result in pool.imap_unordered(
        process_batch, [(batch, train_data) for batch in batch_ids]
      ):
        results.append(result)
        pbar.update(1)

  token_counts = merge_counts(results)
  # token_counts_str = {
  #   tokenizer.decode([token_id]): count for token_id, count in token_counts.items()
  # }

  output_file = f"{output_dir}/token_counts.json"
  with open(output_file, "w") as f:
    json.dump(token_counts, f, indent=2)

  logger.info(f"Token counts saved to {output_file}")

  total_tokens = sum(token_counts.values())
  unique_tokens = len(token_counts)
  logger.info(f"Total tokens: {total_tokens}")
  logger.info(f"Unique tokens: {unique_tokens}")
  logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
  logger.info("\n10 most common tokens:")
  for token, count in Counter(token_counts).most_common(10):
    logger.info(f"{token}: {count}")


if __name__ == "__main__":
  config: CorpusCountConfig = OmegaConf.merge(
    OmegaConf.structured(CorpusCountConfig), OmegaConf.from_cli()
  )

  num_proc = mp.cpu_count() // 2

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  get_stats(config.dataset_name, tokenizer, config.output_dir, config.batch_size, num_proc)
