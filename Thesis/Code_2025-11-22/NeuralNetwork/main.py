"""
This script compares discrete RMSProp optimization with various SDE (Stochastic Differential Equation) 
approximations for neural network training. It supports both ballistic and batch equivalent regimes.
"""

import argparse
import os
import time
from typing import Dict, Tuple, Optional, Any

import torch
import torchsde
import wandb
import math
from torchinfo import summary


from Algorithms.Utils import get_regime_functions
from NeuralNetwork.Utils import set_seed, process_results, get_parameters, log_results_on_wandb
from NeuralNetwork.Dnn import ShallowNN, MLP, ResNet

import sys
sys.setrecursionlimit(100000)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive parameter configuration."""
    parser = argparse.ArgumentParser(
        description='Neural Network Training with RMSProp SDE Approximations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, choices=['ShallowNN', 'MLP', 'ResNet'], default='MLP', help='Neural network model to use')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--c', type=float, default=0.5, help='RMSProp scaling constant of beta')
    train_group.add_argument('--c-1', type=float, default=1, help='C 1 parameter for Adam optimizer')
    train_group.add_argument('--c-2', type=float, default=0.5, help='C 2 parameter for Adam optimizer')
    train_group.add_argument('--epsilon', type=float, default=0.1, help='Regularization epsilon for RMSProp')
    train_group.add_argument('--skip-initial-point', type=int, default=1, help='Number of initial points to skip in analysis')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run simulations on (cpu or cuda)')
    # train_group.add_argument('--device', type=str, default='cpu', help='Device to run simulations on (cpu or cuda)')
    train_group.add_argument('--batch-size', type=int, default=16, help='Batch size for training')

    # Regime selection
    regime_group = parser.add_argument_group('Regime Configuration')
    regime_group.add_argument('--regime', type=str, choices=['balistic', 'batch_equivalent'], default='balistic', help='Optimization regime to use')
    regime_group.add_argument('--simulations', type=str, nargs='+', choices=['1st_order_sde', '2nd_order_sde'], default=['1st_order_sde'], help='Types of simulations to run')
    regime_group.add_argument('--optimizer', type=str, choices=['Adam', 'RMSProp'], default='Adam', help='Optimizer to use for discrete simulations')

    # Random seeds
    seed_group = parser.add_argument_group('Random Seeds')
    seed_group.add_argument('--seed-disc', type=int, default=100, help='Random seed for discrete simulations')
    seed_group.add_argument('--seed-1st', type=int, default=150, help='Random seed for 1st order SDE simulations')
    seed_group.add_argument('--seed-2nd', type=int, default=200, help='Random seed for 2nd order SDE simulations')
    seed_group.add_argument('--seed-parameters', type=int, default=50, help='Random seed for initial parameters')
    seed_group.add_argument('--seed-noise', type=int, default=25, help='Random seed for noise generation')

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--results-dir', type=str, default='results', help='Base directory for saving results')
    output_group.add_argument('--verbose', type=bool, default=True, help='Enable verbose output')
    output_group.add_argument('--wandb', type=bool, default=True, help='Enable Weights & Biases logging')

    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--test-size', type=float, default=0.2, help='Fraction of dataset to use for validation')
    data_group.add_argument('--random-state', type=int, default=42, help='Random state for train/test split')

    # Checkpoint configuration
    checkpoint_group = parser.add_argument_group('Checkpoint Configuration')
    checkpoint_group.add_argument('--checkpoint', type=torch.Tensor, default=torch.linspace(0, 1, 11), help='Checkpoints for recording metrics during training')

    return parser.parse_args()

def run_sde_simulations(
        model_factory: callable,
        optimizer: str,
        regime_funcs: Dict[str, Any],
        which_approximation: str,
        y0: torch.Tensor,
        tau: float,
        c: float,
        final_time: float,
        skip_initial_point: int,
        initial_params: torch.Tensor,
        dim_weights: int,
        num_runs: int,
        batch_size: int,
        epsilon: float,
        sigma_value: float,
        checkpoint: torch.Tensor,
        seed: int,
        device: str = 'cpu',
        verbose: bool = False
    ) -> Dict[str, Any]:
    print(f"[{which_approximation}] Starting continuous simulations...")
    set_seed(seed)

    t0 = time.time()
    ts = torch.arange(tau * skip_initial_point, final_time, tau).to(device)
    num_steps = len(ts)
    num_batched_runs = math.ceil(num_runs / batch_size)
    
    if optimizer == 'Adam':
        dim_result = dim_weights * 3  # theta, m, v
    elif optimizer == 'RMSProp':
        dim_result = dim_weights * 2  # theta, v
    loss, runs = torch.zeros(num_batched_runs, batch_size, num_steps), torch.zeros(num_batched_runs, batch_size, num_steps, dim_result), 
    model = model_factory()

    y0_batched = y0.unsqueeze(0).expand(batch_size, -1).to(device)

    if which_approximation == 'approx_2_fun':
        # dt = tau**2
        dt = tau
    elif which_approximation == 'approx_1_fun':
        dt = tau

    for run in range(num_batched_runs):
        if verbose:
            print(f"[{which_approximation}] Simulation {run+1}/{num_batched_runs}...")
        
        # Setup regularizer
        regime_funcs['regularizer'].set_costant(tau * torch.ones_like(initial_params))
        
        # Create and run SDE
        sde = regime_funcs[which_approximation](
            tau, c, model, ts, regime_funcs['regularizer'], 
            Verbose=True, epsilon=epsilon, sigma_value=sigma_value
        )

        res_cont = torchsde.sdeint(sde, y0_batched, ts, method='euler', dt=dt).permute(1, 0, 2)
        runs[run] = res_cont

        # Compute validation loss with batch processing
        for t in range(num_steps):
            loss[run, : , t] = model.loss_batch(res_cont[:, t, :dim_weights])
    
    # Aggregate results
    loss = loss.reshape(-1, loss.shape[2])
    res_loss = process_results(loss, checkpoints = checkpoint, variable='Loss', Norm_bool=False)
    res = res_loss

    runs =  runs.reshape(-1, runs.shape[2], runs.shape[3])
    res_theta = process_results(runs[:, :, :dim_weights], checkpoints = checkpoint, variable='theta')
    res.update(res_theta)
    if optimizer == 'Adam':
        res_v = process_results(runs[:, :, 2 * dim_weights:], checkpoints = checkpoint, variable='v')
    elif optimizer == 'RMSProp':
        res_v = process_results(runs[:, :, dim_weights:], checkpoints = checkpoint, variable='v')
    res.update(res_v)
    if optimizer == 'Adam':
        res_m = process_results(runs[:, :, dim_weights: 2*dim_weights], checkpoints = checkpoint, variable='m')
        res.update(res_m)

    t1 = time.time()

    additional_info =  {
        'time_steps': ts.cpu(),
        'initial_point': y0.to('cpu'),
        'time_elapsed': t1 - t0,
        'n_runs': num_batched_runs * batch_size,
        'checkpoints': checkpoint.to('cpu')
    }
    res.update(additional_info)
    
    print(f"[{which_approximation}] Simulations completed.\n")
    del model, runs, loss, y0_batched, res_cont, sde
    torch.cuda.empty_cache()
    return res

def run_discrete_simulations(
    model_factory: callable,
    optimizer: str,
    regime_funcs: Dict[str, Any],
    noise: torch.Tensor,
    tau: float,
    beta: float,
    c: float,
    num_steps: int,
    initial_params: torch.Tensor,
    skip_initial_point: int,
    epsilon: float,
    num_runs: int,
    batch_size: int,
    dim_weights: int,
    checkpoint: torch.Tensor,
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run discrete RMSProp simulations with batch processing.
    
    Returns:
        Dict of (Loss_disc, theta_mean_disc, v_mean_disc, y0)
    """
    print("[DISCRETE] Starting discrete simulations...")
    set_seed(seed)
    t0 = time.time()
    num_batched_runs = math.ceil(num_runs / batch_size)
    
    if optimizer == 'Adam':
        dim_result = dim_weights * 3  # theta, m, v
    elif optimizer == 'RMSProp':
        dim_result = dim_weights * 2  # theta, v
    Loss_disc, discrete_runs = torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point),  torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point, dim_result)
    model = model_factory()
    y0 = None
    
    for run in range(num_batched_runs):
        if verbose:
            print(f"[DISCRETE] Simulation {run+1}/{num_batched_runs}...")

        res, loss_values_disc = regime_funcs['discr_fun'](
            model, noise, tau, beta, c, num_steps, initial_params.unsqueeze(0).expand(batch_size, -1),
            skip_initial_point, epsilon=epsilon
        )
        
        Loss_disc[run] = loss_values_disc[:, skip_initial_point:]
        discrete_runs[run] = res[:, skip_initial_point:, :]
        
        if y0 is None:
            y0 = res[0, skip_initial_point, :]

    # Aggregate results
    discrete_runs = discrete_runs.reshape(-1, discrete_runs.shape[2], discrete_runs.shape[3])
    Loss_disc = Loss_disc.reshape(-1, Loss_disc.shape[2])
    res_loss = process_results(Loss_disc, checkpoints = checkpoint, variable='Loss', Norm_bool=False)
    res = res_loss
    
    res_theta = process_results(discrete_runs[:, :, :dim_weights], checkpoints = checkpoint, variable='theta')
    if optimizer == 'RMSProp':
        res_v = process_results(discrete_runs[:, :, dim_weights:], checkpoints = checkpoint, variable='v')
    elif optimizer == 'Adam':
        res_v = process_results(discrete_runs[:, :, 2 * dim_weights:], checkpoints = checkpoint, variable='v')
    res.update(res_theta)
    res.update(res_v)
    if optimizer == 'Adam':
        res_m = process_results(discrete_runs[:, :, dim_weights: 2*dim_weights], checkpoints = checkpoint, variable='m')
        res.update(res_m)

    t1 = time.time()
    additional_info = {
        'initial_point': y0.to('cpu'),
        'time_steps': torch.arange(tau * skip_initial_point, tau * (num_steps), tau).cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': num_batched_runs * batch_size,
        'checkpoints': checkpoint.to('cpu')
    }
    res.update(additional_info)

    print("[DISCRETE] Simulations completed.\n")
    del model, loss_values_disc, discrete_runs, Loss_disc
    torch.cuda.empty_cache()
    return res

def run_1st_order_sde_simulations(
    regime: str,
    optimizer: str,
    model_factory: callable,
    regime_funcs: Dict[str, Any],
    y0: torch.Tensor,
    tau: float,
    c: float,
    final_time: float,
    skip_initial_point: int,
    initial_params: torch.Tensor,
    dim_weights: int,
    num_runs: int,
    batch_size: int,
    epsilon: float,
    sigma_value: float,
    checkpoint: torch.Tensor,
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run 1st order SDE simulations (regime-dependent).
    
    Returns:
        Tuple of results, some may be None depending on regime
    """
    ts = torch.arange(tau * skip_initial_point, final_time, tau).to(initial_params.device)
    
    if regime == 'balistic':
        return _run_1st_order_balistic(
            model_factory, optimizer, regime_funcs, y0, tau, c, final_time, skip_initial_point, ts, initial_params,
            dim_weights, num_runs, batch_size, epsilon, sigma_value, checkpoint, seed, device, verbose=verbose
        )
    elif regime == 'batch_equivalent':
        return run_sde_simulations(
            model_factory, optimizer, regime_funcs, 'approx_1_fun', y0, tau, c, final_time, skip_initial_point,
            initial_params, dim_weights, num_runs, batch_size, epsilon, sigma_value, checkpoint, seed, device, verbose=verbose
        ), None

def _run_1st_order_balistic(
    model_factory: callable,
    optimizer: str,
    regime_funcs: Dict[str, Any],
    y0: torch.Tensor,
    tau: float,
    c: float,
    final_time: float,
    skip_initial_point: int,
    ts: torch.Tensor,
    initial_params: torch.Tensor,
    dim_weights: int,
    num_runs: int,
    batch_size: int,
    epsilon: float,
    sigma_value: float,
    checkpoint: torch.Tensor,
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run 1st order SDE simulations for ballistic regime with batch processing."""
    print("[1ST ORDER SDE] Starting deterministic simulation (1st order)...")
    t0 = time.time()

    # Deterministic simulation
    model = model_factory()
    regime_funcs['regularizer'].set_costant(tau * torch.ones_like(initial_params))

    sde1 = regime_funcs['approx_1_fun_det'](
        tau, c, model, ts, regime_funcs['regularizer'],
        Verbose=verbose, epsilon=epsilon, sigma_value=sigma_value
    )
    
    res_cont_1 = torchsde.sdeint(sde1, y0.unsqueeze(0), ts, method='euler', dt=tau).permute(1, 0, 2)
    
    # Compute losses for deterministic with batch processing
    theta_batch = res_cont_1[0, :, :dim_weights].to(device)
    loss_1_order = model.loss_batch(theta_batch)
    res_loss = process_results(loss_1_order.unsqueeze(0), checkpoints = checkpoint, variable='Loss', Norm_bool=False)
    res = res_loss

    res_theta = process_results(res_cont_1[:, :, :dim_weights], checkpoints = checkpoint, variable='theta')
    if optimizer == 'RMSProp':
        res_v = process_results(res_cont_1[:, :, dim_weights:], checkpoints = checkpoint, variable='v')
    elif optimizer == 'Adam':
        res_v = process_results(res_cont_1[:, :, 2 * dim_weights:], checkpoints = checkpoint, variable='v')
    res.update(res_theta)
    res.update(res_v)
    if optimizer == 'Adam':
        res_m = process_results(res_cont_1[:, :, dim_weights:2*dim_weights], checkpoints = checkpoint, variable='m')
        res.update(res_m)
    t1 = time.time()
    additional_info = {
        'time_steps': ts.cpu(),
        'time_elapsed': t1 - t0,
        'checkpoints': checkpoint.to('cpu'),
        'n_runs': 1
    }
    res.update(additional_info)
    print("[1ST ORDER SDE] Deterministic simulation completed.")
    
    del model, sde1, res_cont_1, theta_batch, loss_1_order
    torch.cuda.empty_cache()

    # Stochastic simulations
    res_1_order_stoc = run_sde_simulations(
            model_factory, optimizer, regime_funcs, 'approx_1_fun', y0, tau, c, final_time, skip_initial_point,
            initial_params, dim_weights, num_runs, batch_size, epsilon, sigma_value, checkpoint, seed, device, verbose=verbose
        )
    
    print("[1ST ORDER SDE] Stochastic simulations completed.\n")
    
    return res_1_order_stoc, res

def run_experiment_configuration(
    args: argparse.Namespace,
    model_factory: callable,
    tau: float,
    sigma_value: float,
    checkpoint: torch.Tensor,
    epsilon: float = 0.1
) -> None:
    """
    Run a complete experiment configuration for given hyperparameters.
    
    Returns:
        Tuple of (max_err_1st, max_err_2nd)
    """

    # Setup experiment
    if args.regime == 'balistic':
        regime_name = 'Balistic'
    elif args.regime == 'batch_equivalent':
        regime_name = 'BatchEq'

    if args.optimizer == 'Adam':
        args.c = (args.c_1, args.c_2)
        beta = (1 - tau * args.c_1, 1 - tau * args.c_2)
        print(f"\n==================== tau = {tau}, C1 = {args.c_1}, C2 = {args.c_2}, BETA = {beta}, SIGMA = {sigma_value}, epsilon {epsilon} ====================\n")
    elif args.optimizer == 'RMSProp':
        beta = 1 - tau * args.c
        print(f"\n==================== tau = {tau}, C = {args.c}, BETA = {beta}, SIGMA = {sigma_value}, epsilon {epsilon} ====================\n")

    # Create result directory
    result_dir = f"{args.results_dir}_tau_{tau}_c_{args.c}_sigma_{sigma_value}_finaltime_{args.final_time}"
    result_dir = os.path.join(result_dir, args.regime.replace(' ', '_'))
    os.makedirs(result_dir, exist_ok=True)
    
    # Get initial parameters
    set_seed(args.seed_parameters)
    initial_model = model_factory()
    initial_params = initial_model.initial_weights
    dim_weights = initial_params.shape[0]
    del initial_model
    
    # Setup time parameters
    num_steps = int(torch.ceil(torch.tensor(args.final_time / tau)).item())
    print(f'Number of steps: {num_steps}')
    
    # Get regime functions
    regime_funcs = get_regime_functions(args.regime, args.optimizer)
    
    # Generate noise
    set_seed(args.seed_noise)
    noise = sigma_value * torch.randn((5000, initial_params.shape[0]))
        
    # Run discrete simulations
    res_disc = run_discrete_simulations(
        model_factory, args.optimizer, regime_funcs, noise, tau, beta, args.c, num_steps, 
        initial_params, args.skip_initial_point, epsilon, args.num_runs, args.batch_size,
        dim_weights, checkpoint, args.seed_disc, args.device, args.verbose
    )
    base_results = {
        'disc': res_disc,
        'final_time': args.final_time,
        'tau': tau,
        'c': args.c,
        'sigma': sigma_value,
        'epsilon': epsilon,
        'regime': args.regime,
        'optimizer': args.optimizer,
        'skipped_initial_points': args.skip_initial_point,
        'simulation keys': ['disc']
    }
    log_results_on_wandb(base_results, args, 'disc', tau, sigma_value, result_dir)

    y0 = res_disc['initial_point'].to(args.device)
    # Run 1st order SDE simulations
    res_1_order_det = None
    if '1st_order_sde' in args.simulations:
        res_1_order_stoc, res_1_order_det = run_1st_order_sde_simulations(
            args.regime, args.optimizer, model_factory, regime_funcs, y0, tau, args.c, args.final_time,
            args.skip_initial_point, initial_params, dim_weights,
            args.num_runs, args.batch_size, epsilon, sigma_value, checkpoint, args.seed_1st, args.device, args.verbose
        )
        base_results['1_order_stoc'] = res_1_order_stoc
        base_results['simulation keys'] += ['1_order_stoc']
        log_results_on_wandb(base_results, args, '1_order_stoc', tau, sigma_value, result_dir)
        if res_1_order_det is not None:
            base_results['1_order_det'] = res_1_order_det
            base_results['simulation keys'] += ['1_order_det']
            log_results_on_wandb(base_results, args, '1_order_det', tau, sigma_value, result_dir)

    # # Run 2nd order SDE simulations
    if '2nd_order_sde' in args.simulations:
        res_2_order = run_sde_simulations(
            model_factory, args.optimizer, regime_funcs, 'approx_2_fun', y0, tau, args.c, args.final_time,
            args.skip_initial_point, initial_params, dim_weights,
            args.num_runs, args.batch_size, epsilon, sigma_value, checkpoint, args.seed_2nd, args.device, args.verbose
        )
        base_results['2_order_stoc'] = res_2_order
        base_results['simulation keys'] += ['2_order_stoc']
        log_results_on_wandb(base_results, args, '2_order_stoc', tau, sigma_value, result_dir)
    

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()    
    
    if args.model == 'ShallowNN':
        def model_factory():
            return ShallowNN()
    elif args.model == 'MLP':
        def model_factory():
            return MLP()
    elif args.model == 'ResNet':
        def model_factory():
            return ResNet(device=args.device)
        summary(model_factory().network, input_size=(1, 1, 28, 28))
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    tau, sigma, final_time, num_runs, batch_size, c, c1, c2 = get_parameters(args.model)
    args.num_runs = num_runs
    args.final_time = final_time
    args.batch_size = batch_size
    args.tau = tau

    if c is not None:
        args.c = c
    if c1 is not None:
        args.c_1 = c1
    if c2 is not None:
        args.c_2 = c2



    print("Starting Neural Network Training with RMSProp SDE Approximations")
    print(f"Optimizer: {args.optimizer}, Regime: {args.regime}, Model: {args.model}")
    print(f"Parameters: tau={tau}, c={args.c}, sigma={sigma}, final_time={args.final_time}")
    print(f"Number of runs: {args.num_runs}")

    if isinstance(sigma, list):
        for s in sigma:
            args.sigma = s
            run_experiment_configuration(
                args, model_factory, tau, s, checkpoint = args.checkpoint, epsilon = args.epsilon
            )
    else:
        args.sigma = sigma
        run_experiment_configuration(
            args, model_factory, tau, sigma, checkpoint = args.checkpoint, epsilon = args.epsilon
        )
    
    print("All experiments completed successfully!")



if __name__ == "__main__":
    main()


'''
python -m NeuralNetwork.main_v3 --regime balistic --optimizer RMSProp; python -m NeuralNetwork.main_v3 --regime balistic --optimizer Adam; python -m NeuralNetwork.main_v3 --regime batch_equivalent --optimizer Adam; python -m NeuralNetwork.main_v3 --regime batch_equivalent --optimizer RMSProp;

'''
