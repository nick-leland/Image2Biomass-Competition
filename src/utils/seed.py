"""
Seed utilities for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # PyTorch backend settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed} for reproducibility")


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader to ensure reproducibility.

    Args:
        worker_id: ID of the DataLoader worker

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
