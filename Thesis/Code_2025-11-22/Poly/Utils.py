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


def get_regime_functions(regime: str, optimizer: str) -> Dict[str, Any]:
    """
    Import and return the appropriate functions for the selected regime.
    
    Args:
        regime: Either 'balistic' or 'batch equivalent'
        optimizer: The optimizer to use ('Adam' or 'RMSProp')
        
    Returns:
        Dictionary containing regime-specific functions
    """
    if regime == 'balistic':
        if optimizer == 'Adam':
            from Balistic_regime_Adam import (
                Discrete_Adam_balistic_regime,
                Adam_SDE_2order_balistic_regime,
                Adam_deterministic,
                Regularizer_ReLu,
                Adam_SDE_1order_balistic_regime
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_Adam_balistic_regime,
                'approx_1_fun_det': Adam_deterministic,
                'approx_2_fun': Adam_SDE_2order_balistic_regime,
                'approx_1_fun': Adam_SDE_1order_balistic_regime
            }
        elif optimizer == 'RMSProp':
            from Balistic_regime_RMSProp import (
                Discrete_RMProp_balistic_regime,
                RMSprop_SDE_2order_balistic_regime,
                RMSprop_deterministic,
                Regularizer_ReLu,
                RMSprop_SDE_1order_balistic_regime
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_RMProp_balistic_regime,
                'approx_1_fun_det': RMSprop_deterministic,
                'approx_2_fun': RMSprop_SDE_2order_balistic_regime,
                'approx_1_fun': RMSprop_SDE_1order_balistic_regime
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    elif regime == 'batch_equivalent':
        if optimizer == 'Adam':
            from Batch_eq_regime_Adam import (
                Discrete_Adam_batch_equivalent_regime,
                Adam_SDE_1order_batch_equivalent_regime,
                Adam_SDE_2order_batch_equivalent_regime,
                Regularizer_ReLu
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_Adam_batch_equivalent_regime,
                'approx_1_fun': Adam_SDE_1order_batch_equivalent_regime,
                'approx_2_fun': Adam_SDE_2order_batch_equivalent_regime
            }
        elif optimizer == 'RMSProp':
            from Batch_eq_regime_RMSProp import (
                Discrete_RMProp_batch_eq_regime,
                RMSprop_SDE_1order_batch_eq_regime,
                RMSprop_SDE_2order_batch_eq_regime,
                Regularizer_ReLu
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_RMProp_batch_eq_regime,
                'approx_1_fun': RMSprop_SDE_1order_batch_eq_regime,
                'approx_2_fun': RMSprop_SDE_2order_batch_eq_regime
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    else:
        raise ValueError(f"Unknown regime: {regime}")
    



def save_results(
    result_dir: str,
    regime_name: str,
    Loss_disc: torch.Tensor,
    loss_1_order_stoc: torch.Tensor,
    loss_2_order: torch.Tensor,
    Val_loss_disc: torch.Tensor,
    val_loss_1_order_stoc: torch.Tensor,
    val_loss_2_order: torch.Tensor,
    loss_1_order: Optional[torch.Tensor] = None,
    val_loss_1_order: Optional[torch.Tensor] = None,
    final_distribution_1_order_det: Optional[torch.Tensor] = None,
    final_distribution_1_order_stoc: Optional[torch.Tensor] = None,
    final_distribution_disc: Optional[torch.Tensor] = None,
    final_distribution_2_order: Optional[torch.Tensor] = None
) -> None:
    """Save numerical results to .npy files."""
    print(f"Saving results to {result_dir}...")
    
    # Save loss results
    np.save(os.path.join(result_dir, f'mean_loss_disc_{regime_name}.npy'), 
            Loss_disc.cpu().detach().numpy())
    np.save(os.path.join(result_dir, f'mean_loss_1st_{regime_name}.npy'), 
            loss_1_order_stoc.cpu().detach().numpy())
    np.save(os.path.join(result_dir, f'mean_loss_2nd_{regime_name}.npy'), 
            loss_2_order.cpu().detach().numpy())
    
    # Save validation loss results
    np.save(os.path.join(result_dir, f'mean_val_loss_disc_{regime_name}.npy'), 
            Val_loss_disc.cpu().detach().numpy())
    np.save(os.path.join(result_dir, f'mean_val_loss_1st_{regime_name}.npy'), 
            val_loss_1_order_stoc.cpu().detach().numpy())
    np.save(os.path.join(result_dir, f'mean_val_loss_2nd_{regime_name}.npy'), 
            val_loss_2_order.cpu().detach().numpy())
    
    # Save stochastic results if available
    if loss_1_order is not None:
        np.save(os.path.join(result_dir, f'mean_loss_stoc_1st_{regime_name}.npy'), 
                loss_1_order.cpu().detach().numpy())
    if val_loss_1_order is not None:
        np.save(os.path.join(result_dir, f'mean_val_loss_stoc_1st_{regime_name}.npy'), 
                val_loss_1_order.cpu().detach().numpy())
        
    # Save final distributions if available
    if final_distribution_1_order_det is not None:
        np.save(os.path.join(result_dir, f'final_distribution_1st_det_{regime_name}.npy'), 
                final_distribution_1_order_det.cpu().detach().numpy())
    if final_distribution_1_order_stoc is not None:
        np.save(os.path.join(result_dir, f'final_distribution_1st_stoc_{regime_name}.npy'), 
                final_distribution_1_order_stoc.cpu().detach().numpy())
    if final_distribution_disc is not None:
        np.save(os.path.join(result_dir, f'final_distribution_disc_{regime_name}.npy'), 
                final_distribution_disc.cpu().detach().numpy())
    if final_distribution_2_order is not None:
        np.save(os.path.join(result_dir, f'final_distribution_2nd_{regime_name}.npy'), 
                final_distribution_2_order.cpu().detach().numpy())