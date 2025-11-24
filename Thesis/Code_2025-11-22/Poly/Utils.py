import os
import random
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def norm_and_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor and compute mean across batches.
    
    Args:
        tensor: Input tensor of shape (num_batch, num_steps, num_params)
        
    Returns:
        Mean norm across batches
    """
    norm = torch.norm(tensor, dim=2, keepdim=True)
    mean = torch.mean(norm, dim=0)
    final_norm = norm[:, -1, :]
    return mean, final_norm