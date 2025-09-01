# %%
from datasets import load_dataset
from transformers import GPTNeoXTokenizerFast, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing

from omegaconf import OmegaConf
from ood_prompts.config import TokenizerTrainConfig
from ood_prompts.utils import get_logger


logger = get_logger(__name__)

BOS_TOK = "<|startoftext|>"
UNK_TOK = "<|unk|>"
EOS_TOK = "<|endoftext|>"


def text_iterator(dataset):
  for example in dataset:
    yield example["story"]


if __name__ == "__main__":
  config: TokenizerTrainConfig = OmegaConf.merge(
    OmegaConf.structured(TokenizerTrainConfig), OmegaConf.from_cli()
  )

  dataset = load_dataset(config.dataset_path, split="train")
  tokenizer_gptneox = GPTNeoXTokenizerFast.from_pretrained(config.base_model_name)

  logger.info(f"Training tokenizer on {config.dataset_path} with type '{config.type}'")

  if config.type == "bpe":
    tokenizer_gptneox.pad_token = tokenizer_gptneox.eos_token

    tokenizer = tokenizer_gptneox.train_new_from_iterator(
      text_iterator(dataset), vocab_size=config.vocab_size
    )
    tokenizer.save_pretrained(config.save_path)
  else:
    logger.info(f"Training word-level tokenizer based on {config.base_model_name}")
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOK))
    tokenizer.pre_tokenizer = Sequence([Punctuation(), Whitespace()])

    trainer = WordLevelTrainer(
      vocab_size=256_000,  # make the vocab big enough to handle all possible words, usually around ~50k unique
      min_frequency=1,
      special_tokens=[BOS_TOK, EOS_TOK, UNK_TOK],
    )
    tokenizer.train_from_iterator(text_iterator(dataset), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
      single=f"{BOS_TOK} $A", special_tokens=[(BOS_TOK, tokenizer.token_to_id(BOS_TOK))]
    )

    wrapped_tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=tokenizer,
      unk_token=UNK_TOK,
      bos_token=BOS_TOK,
      eos_token=EOS_TOK,
      pad_token=EOS_TOK,
    )
    wrapped_tokenizer.save_pretrained(config.save_path)

  logger.info(f"Tokenizer saved to {config.save_path}")
