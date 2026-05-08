import os

import torch


def configure_cpu_threads(default_threads=8):
    threads = int(os.environ.get("TORCH_NUM_THREADS", str(default_threads)))
    torch.set_num_threads(max(1, threads))
    return torch.get_num_threads()
