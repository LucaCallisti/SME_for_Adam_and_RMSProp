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



def load_and_preprocess_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the California Housing dataset.
    
    Args:
        test_size: Fraction of data to use for validation
        random_state: Random state for reproducible splits
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val) as torch tensors
    """
    print("Loading and preprocessing California Housing dataset...")
    
    # Load dataset
    data = fetch_california_housing()
    X = data['data']
    y = data['target'].reshape(-1, 1)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature scaling
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    
    # Target scaling
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    return X_train, X_val, y_train, y_val
