import os
import random
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
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



def load_and_preprocess_data(dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess datasets (Housing, BreastCancer, MNIST, CIFAR10).
    
    Args:
        test_size: Fraction of data to use for validation
        random_state: Random state for reproducible splits
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val) as torch tensors
    """
    print(f"Loading and preprocessing {dataset} dataset...")
    
    if dataset == 'Housing':
        data = fetch_california_housing()
        X = data['data']
        y = data['target']
    elif dataset == 'BreastCancer':
        data = load_breast_cancer()
        X = data['data']
        y = data['target']
    elif dataset == 'MNIST':
        data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = data['data']
        y = data['target'].astype(int)
    elif dataset == 'CIFAR10':
        train_set = CIFAR10(root='./data', train=True, download=True)
        test_set = CIFAR10(root='./data', train=False, download=True)
        
        X = np.concatenate([train_set.data, test_set.data], axis=0) # Shape: (60000, 32, 32, 3)
        y = np.concatenate([train_set.targets, test_set.targets], axis=0)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y = np.array(y).reshape(-1, 1)
    
    if dataset == 'MNIST':
        X = X / 255.0 # Normalize [0, 1]
        X = X.reshape(-1, 28, 28)  
        X = np.expand_dims(X, axis=1)

        mean = 0.1307
        std = 0.3081
        X = (X - mean) / std

    elif dataset == 'CIFAR10':
        # (N, 32, 32, 3) -> (N, 3, 32, 32)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X / 255.0 # Normalize [0, 1]
        
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        X = (X - mean) / std

    stratify = y if dataset in ['BreastCancer', 'MNIST', 'CIFAR10'] else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    if dataset in ['Housing', 'BreastCancer']:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
    
    if dataset == 'Housing':
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
    
    return X_train, X_val, y_train, y_val

def get_parameters(model_type):
    if model_type == 'ShallowNN':
        tau = 0.001
        sigma = 1
        final_time = 50
        num_runs = 32
    elif model_type == 'MLP':
        tau = 0.001
        sigma = 1
        final_time = 50
        num_runs = 32
    elif model_type == 'ResNet':
        tau = 0.001
        sigma = 1
        final_time = 5
        num_runs = 32
    return tau, sigma, final_time, num_runs