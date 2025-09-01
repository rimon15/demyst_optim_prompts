import time
import os
from datetime import datetime
import logging
from functools import wraps
import gc
import torch


def split_data_among_gpus(data: list[str], gpu_idx: int, n_gpus: int) -> tuple[list, int]:
  total_data_len = len(data)
  chunk_size = total_data_len // n_gpus
  remainder = total_data_len % n_gpus

  if gpu_idx < remainder:
    start_idx = gpu_idx * (chunk_size + 1)
    end_idx = start_idx + chunk_size + 1
  else:
    start_idx = gpu_idx * chunk_size + remainder
    end_idx = start_idx + chunk_size

  return data[start_idx:end_idx], start_idx


def cleanup_gpu_memory():
  gc.collect()
  torch.cuda.empty_cache()


def get_top_k_predictions(logits: torch.Tensor, k=10) -> set[int]:
  return set(logits.topk(k).indices.tolist())


def calculate_overlap(set1: set, set2: set) -> float:
  return len(set1.intersection(set2)) / len(set1)


class Timer:
  def __init__(
    self,
    desc: str = "",
    fpath: str | None = None,
    logger: logging.Logger | None = None,
    do_print: bool = True,
  ):
    self.desc = desc
    self.fpath = fpath
    self.logger = logger
    self.do_print = do_print

  def __enter__(self):
    self.start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end_time = time.time()
    elapsed = self.end_time - self.start_time

    if self.do_print:
      print(f"{self.desc} took: {elapsed} seconds")
    if self.logger is not None:
      self.logger.info(f"{self.desc} took: {elapsed} seconds")

    if self.fpath is not None:
      with open(self.fpath, "a+") as f:
        f.write(f"{self.desc}:\t{elapsed}\n")

  def __call__(self, func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      with self:
        return func(*args, **kwargs)

    return wrapper


def get_logger(module_name: str) -> logging.Logger:
  current_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(current_dir, ".."))
  logs_dir = os.path.join(project_root, "logs")

  os.makedirs(logs_dir, exist_ok=True)
  log_filename = os.path.join(logs_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

  logger = logging.getLogger(module_name)
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

  file_handler = logging.FileHandler(log_filename)
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  return logger
