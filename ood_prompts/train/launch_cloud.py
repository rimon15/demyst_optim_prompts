# Twins/suffix optim launching for skypilot
# %%
import sky

from omegaconf import OmegaConf
import asyncio
import concurrent.futures
import json

from ood_prompts.config import CloudTrainConfig
from ood_prompts.utils import get_logger


logger = get_logger(__name__)


def launch_task(task: sky.Task):
  return sky.launch(task, down=True)


async def run_task_async(task: sky.Task):
  loop = asyncio.get_event_loop()
  with concurrent.futures.ThreadPoolExecutor() as pool:
    return await loop.run_in_executor(pool, launch_task, task)


async def run_multiple_tasks(tasks: list[sky.Task]):
  coroutines = [run_task_async(task) for task in tasks]
  return await asyncio.gather(*coroutines)


# %%
if __name__ == "__main__":
  config: CloudTrainConfig = OmegaConf.merge(
    OmegaConf.structured(CloudTrainConfig), OmegaConf.from_cli()
  )

  logger.info(config)

  with open(config.local_data_path, "r") as f:
    prompts = json.load(f)
  total_len = len(prompts)

  # we launch 1 gpu per instance
  tasks = []
  for i in range(config.n_instances):
    task = sky.Task.from_yaml(config.config_path)
    task.name = f"{config.name}_{i}"
    task.workdir = config.workdir

    chunk_size = total_len // config.n_instances
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size if i < config.n_instances - 1 else total_len

    task.update_envs(
      {
        "MODEL_BUCKET": config.model_bucket,
        "MODEL_PATH": config.model_path,
        "DATASET_BUCKET": config.dataset_bucket,
        "DATASET_PATH": config.dataset_path,
        "OUTPUT_BUCKET": config.output_bucket,
        "OUTPUT_PATH": config.output_path,
        "OUTPUT_FOLDER": config.output_folder,
        "TRAIN_TYPE": config.train_type,
        "BATCH_SIZE": config.batch_size,
        "NUM_EPOCHS": config.n_epochs,
        "SUBSET_START": start_idx,
        "SUBSET_END": end_idx,
      }
    )
    tasks.append(task)
    logger.info(f"Created task {i} processing data from index {start_idx} to {end_idx}")

  loop = asyncio.get_event_loop()
  results = loop.run_until_complete(run_multiple_tasks(tasks))

  for i, res in enumerate(results):
    logger.info(f"Task {i} launched with job id: {res}")
