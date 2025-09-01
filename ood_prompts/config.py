from dataclasses import dataclass


@dataclass
class TokenizerTrainConfig:
  dataset_path: str
  save_path: str
  vocab_size: int
  base_model_name: str
  type: str = "word"  # or bpe


@dataclass
class StoriesTrainConfig:
  run_name: str
  base_model_name: str
  tokenizer_path: str
  output_dir: str
  n_train_epochs: int
  dataset_path: str
  dev_percent: float = 0.1
  batch_size: int = 64

  model_max_len: int = 512

  override_default: bool = False
  hidden_size: int = 512
  ffn_dim: int = 2048
  n_layers: int = 6
  n_attn_heads: int = 8

  weight_decay: float = 0.1
  adam_beta1: float = 0.9
  adam_beta2: float = 0.95
  lr: float = 6e-4
  warmup_steps: int = 500
  grad_accum_steps: int = 1
  save_steps: int = 250
  eval_steps: int = 250
  logging_steps: int = 10
  save_total_limit: int = 3


@dataclass
class TwinsTrainConfig:
  model_path: str
  output_path: str

  # (THIS SHOULD ALWAYS BE THE CASE) if we already have prompts ready to go
  prompts_dataset_path: str = ""

  # we dont have any current prompt data so lets generate it: DO NOT USE PARAM THIS, THIS SHOULD PROBABLY NOT BE THE CASE!!! -> WE SHOULD BE USING THE GEN_PROMPT_DATA SCRIPT!!!
  dataset_path: str = ""
  # only for the word-stories
  n_seqs_per_sample: int = 50

  subset_start: int = -1  # if already have the data and want to resume from a certain point
  subset_end: int = -1  # if already have the data and want to resume from a certain point

  gpu_idx: int = 0
  n_gpus: int = 1

  n_inputs: int = 500
  min_sent_words: int = 5
  max_sent_words: int = 15
  n_docs: int = 100
  doc_len: int = 10  # 32
  gen_batch_size: int = 100

  batch_size: int = 100
  early_stop_kl: float = 5.0
  n_epochs: int = 500


@dataclass
class CloudTrainConfig:
  n_instances: int
  local_data_path: str

  config_path: str
  name: str
  workdir: str
  model_bucket: str
  model_path: str
  dataset_bucket: str
  dataset_path: str
  output_bucket: str
  output_path: str
  output_folder: str
  train_type: str = "twins"
  batch_size: int = 8
  n_epochs: int = 500


@dataclass
class CorpusCountConfig:
  dataset_name: str
  tokenizer_path: str
  output_dir: str
  batch_size: int = 10000


@dataclass
class ExtractHiddenConfig:
  model_name: str
  output_dir: str
  base_optim_dir: str

  loss_thresh: float  # only for suffix filtering
  use_suffix: bool = False

  kl_thresh: float = 4.0  # only for twins filtering

  replace_rand: bool = (
    False  # if true, replace the optimized twin with random sequence of tokens from the vocab
  )


@dataclass
class GenPromptDataConfig:
  model_name: str
  dataset_name: str
  out_path: str
  n_prompts: int
  min_prompt_len: int = 5
  max_prompt_len: int = 15


SETUP_LANGUAGES = False
