import os
import random
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any, Optional
import wandb
import math


def log_results_on_wandb(final_results: Dict[str, Any], args: Any, sim : str, tau: float, sigma_value: float, result_dir: str) -> None:
    path = os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}_sigma{sigma_value}.pt')
    torch.save(final_results, path)
   
    effective_runs = math.ceil(args.num_runs / args.batch_size) * args.batch_size

    wandb.init(
        project=f'{args.model}',
        entity = 'Effective-continuous-equations',
        name=f'{args.optimizer}_{args.regime}_{sim}_tau_{tau}_c_{args.c}_sigma_{sigma_value}_nruns_{effective_runs}',
        config=vars(args),
        notes='Comparison of discrete RMSProp with SDE approximations for shallow NN on California Housing dataset with comparison of loss, validation loss, norm of the theta and v and distribution of the final loss and final theta.',
        save_code=True
    )
    artifact = wandb.Artifact(f"final_results_tau_{tau}_sigma_{sigma_value}", type="results")
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    wandb.log({"time_elapsed": final_results[sim]['time_elapsed'], 'runs': effective_runs})


    ts = final_results[sim]['time_steps'].numpy()
    theta = final_results[sim]['theta']
    v = final_results[sim]['v']
    if args.optimizer == 'Adam':
        m = final_results[sim]['m']

    for t in range(len(ts)):
        loss_down = final_results[sim]['Loss'][0, t].item()
        loss_mean = final_results[sim]['Loss'][1, t].item()
        loss_up = final_results[sim]['Loss'][2, t].item()
        theta_down_value = theta[0, t].item()
        theta_mean = theta[1, t].item()
        theta_up_value = theta[2, t].item()
        v_down_value = v[0, t].item()
        v_mean = v[1, t].item()
        v_up_value = v[2, t].item()
        info ={
            f"Loss": loss_mean,
            f"Loss_up": loss_up,
            f"Loss_down": loss_down,
            f"theta": theta_mean,
            f"theta_up": theta_up_value,
            f"theta_down": theta_down_value,
            f"v": v_mean,
            f"v_up": v_up_value,
            f"v_down": v_down_value,
            "time": ts[t]
        }
        if args.optimizer == 'Adam':
            m_down = m[0, t].item()
            m_mean = m[1, t].item()
            m_up = m[2, t].item()
            info.update({
                f"m": m_mean,
                f"m_up": m_up,
                f"m_down": m_down,
            })
        wandb.log(info)
            

    # Log final distributions as histograms to wandb
    def _aux_log_distribution(name: str, data: torch.Tensor, title: str, t : float) -> None:
        data_np = data.cpu().numpy().flatten()
        table = wandb.Table(data=[[v] for v in data_np], columns=["value"])
        wandb.log({
            f"Histogram_{name}/time_{t:.5f}":
                wandb.plot.histogram(table, "value", title=title+f" at time {t:.5f}"),
        },)

    keys = [k if 'distribution' in k else None for k in final_results[sim].keys()]
    for key in keys:
        if key is not None:
            ts = final_results[sim]['time_steps']
            checkpoints = final_results[sim]['checkpoints']
            index_chekpoints = (checkpoints * (ts.shape[0]-1)).long()
            for i, index in enumerate(index_chekpoints):
                _aux_log_distribution(
                    name=key,
                    data=final_results[sim][key][:, i],
                    title=key,
                    t=ts[index]
                )

    wandb.finish()

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


def process_results(tensor: torch.Tensor, checkpoints: torch.Tensor, variable : str, Norm_bool : bool = True) -> torch.Tensor:
    """
    Normalize tensor and compute mean across batches.
    
    Args:
        tensor: Input tensor of shape (num_batch, num_steps, num_params)
        
    Returns:
        Mean norm across batches
    """
    input_ = tensor.clone()
    if Norm_bool:
        tensor = torch.norm(tensor, dim=2, keepdim=True)
    mean = torch.mean(tensor, dim=0)
    std_dev = torch.std(tensor, dim=0)
    if torch.isnan(std_dev).all():
        std_dev = torch.zeros_like(mean)
    mean_down = mean - std_dev
    mean_up = mean + std_dev
    final_result = torch.stack([mean_down.unsqueeze(0), mean.unsqueeze(0), mean_up.unsqueeze(0)], dim=0)
    shapes = tensor.shape[1]
    checkpoints_index = (checkpoints * (shapes - 1)).long()
    distributions_norm = tensor[:, checkpoints_index.squeeze()]

    result = {
        variable: final_result.squeeze(),
        variable + '_distributions': distributions_norm,
        variable + '_mean': mean,
        variable + '_std_dev': std_dev
    }
    return result



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
        train_set = datasets.CIFAR10(root='./data', train=True, download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True)
        
        X = np.concatenate([train_set.data, test_set.data], axis=0) # Shape: (N, 32, 32, 3)
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
    
    if dataset in ['Housing', 'BreastCancer']:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
    
    if dataset == 'Housing':
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)
        
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
    
    return X_train, y_train

def get_parameters(model_type):
    if model_type == 'ShallowNN':
        tau = 0.001
        sigma = [0.01, 0.1, 0.2, 0.5, 0.75, 1.0]
        final_time = 3
        num_runs = 32
        batch_size = 32
        c, c1, c2 = None, None, None
    elif model_type == 'MLP':       # https://arxiv.org/pdf/2411.15958 F.3 RMSprop: SDE validation, DNN on Breast Cancer Dataset
        tau = 0.001
        sigma = 0.01
        final_time = 1.5
        num_runs = 16
        batch_size = 8
        c = 5               # to get beta = 0.9995
        c1 = 100            # to get beta1 = 0.99
        c2 = 5              # to get beta2 = 0.9995      # Nel paper è 0.999 ma lo imposto a 0.9995 per coerenza con beta 
    elif model_type == 'ResNet':
        tau = 0.0001        # Preso da RMSProp
        sigma = 0.0001      # Preso da RMSProp
        final_time = 0.0005    # diverso da loro   # 0.05
        num_runs = 1
        batch_size = 1
        c = 1               # to get beta = 0.9999
        c1 = 100            # to get beta1 = 0.99   
        c2 = 1             # to get beta2 = 0.9999

    return tau, sigma, final_time, num_runs, batch_size, c, c1, c2