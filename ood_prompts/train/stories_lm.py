# %%
# from torch._dynamo.backends import inductor
from transformers import (
  AutoModelForCausalLM,
  GPTNeoXConfig,
  AutoTokenizer,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling,
)
import datasets
from accelerate import Accelerator

from omegaconf import OmegaConf
import os

from ood_prompts.config import StoriesTrainConfig


accelerator = Accelerator()

if __name__ == "__main__":
  config: StoriesTrainConfig = OmegaConf.merge(
    OmegaConf.structured(StoriesTrainConfig), OmegaConf.from_cli()
  )

  model_name = config.base_model_name

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  conf = GPTNeoXConfig.from_pretrained(model_name)
  conf_dict = conf.to_dict()

  conf_dict["max_position_embeddings"] = config.model_max_len

  if config.override_default:
    conf_dict["hidden_size"] = config.hidden_size
    conf_dict["intermediate_size"] = config.ffn_dim
    conf_dict["num_hidden_layers"] = config.n_layers
    conf_dict["num_attention_heads"] = config.n_attn_heads

  conf_dict["vocab_size"] = tokenizer.vocab_size
  # conf_dict["attention_bias"] = False
  # conf_dict["use_parallel_residual"] = False
  # conf_dict["hidden_act"] = "gelu_new"
  model = AutoModelForCausalLM.from_config(GPTNeoXConfig.from_dict(conf_dict))
  print(model)
  print(conf_dict)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"num params: {n_params}")

  # Original tinystories has several issues: https://huggingface.co/datasets/roneneldan/TinyStories/discussions
  dset_name = config.dataset_path

  def tokenize_fn(ex):
    inputs = tokenizer(
      ex["story"] if "GPT4" in dset_name else ex["text"],
      padding="max_length",
      truncation=True,
      max_length=config.model_max_len,
      return_tensors="pt",
    )

    return inputs

  # create/process dataset if not already done
  final_dataset_name = f"{config.output_dir}/dataset"

  if accelerator.is_main_process and not os.path.exists(final_dataset_name):
    dataset = datasets.load_dataset(dset_name)
    tokenized_data = dataset.map(
      tokenize_fn,
      batched=True,
      batch_size=1000,
      num_proc=40,
    )["train"]
    tokenized_data.set_format(type="torch")

    final_dataset = tokenized_data.train_test_split(test_size=config.dev_percent, shuffle=True)

    print(f"Train set size: {len(final_dataset['train'])}")
    print(f"Dev set size: {len(final_dataset['test'])}")

    # save the datasets
    # Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    final_dataset.save_to_disk(f"{config.output_dir}/dataset", max_shard_size="500MB")
  accelerator.wait_for_everyone()

  final_dataset = datasets.load_from_disk(f"{config.output_dir}/dataset")

  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

  train_args = TrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.batch_size,
    do_train=True,
    do_eval=True,
    save_safetensors=True,
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    logging_steps=config.logging_steps,
    eval_steps=config.eval_steps,
    eval_strategy="steps",
    gradient_accumulation_steps=config.grad_accum_steps,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    warmup_steps=config.warmup_steps,
    learning_rate=config.lr,
    weight_decay=config.weight_decay,
    adam_beta1=config.adam_beta1,
    adam_beta2=config.adam_beta2,
    num_train_epochs=config.n_train_epochs,
    logging_dir=f"{config.output_dir}/logs",
    torch_compile=True,
    torch_compile_backend="inductor",
    run_name=config.run_name,
    report_to="wandb",
  )
  trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"],
    data_collator=data_collator,
  )

  trainer.train()
  tokenizer.save_pretrained(config.output_dir)
