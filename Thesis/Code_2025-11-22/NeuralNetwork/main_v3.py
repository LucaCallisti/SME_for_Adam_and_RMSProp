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

from Algorithms.Utils import get_regime_functions
from NeuralNetwork.Utils import set_seed, norm_and_mean, load_and_preprocess_data
from NeuralNetwork.Dnn import ShallowNN

import sys
sys.setrecursionlimit(10000)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive parameter configuration."""
    parser = argparse.ArgumentParser(
        description='Neural Network Training with RMSProp SDE Approximations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--input-dim', type=int, default=None, help='Input dimension (auto-detected from dataset if None)')
    model_group.add_argument('--mid-dim', type=int, default=3, help='Hidden layer dimension')
    model_group.add_argument('--output-dim', type=int, default=1, help='Output dimension')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--tau-list', type=float, nargs='+', default=[0.1], help='Learning rate values to test')
    train_group.add_argument('--c', type=float, default=0.5, help='RMSProp scaling constant of beta')
    train_group.add_argument('--c-1', type=float, default=1, help='C 1 parameter for Adam optimizer')
    train_group.add_argument('--c-2', type=float, default=0.5, help='C 2 parameter for Adam optimizer')
    train_group.add_argument('--sigma-list', type=float, nargs='+', default=[0.2], help='Noise variance values to test')
    train_group.add_argument('--num-runs', type=int, default=128, help='Number of simulation runs for averaging')
    train_group.add_argument('--final-time', type=float, default=10_000.0, help='Final time for SDE integration')
    train_group.add_argument('--epsilon', type=float, default=0.1, help='Regularization epsilon for RMSProp')
    train_group.add_argument('--skip-initial-point', type=int, default=2, help='Number of initial points to skip in analysis')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run simulations on (cpu or cuda)')
    train_group.add_argument('--batch-size', type=int, default=128, help='Batch size for training')

    # Regime selection
    regime_group = parser.add_argument_group('Regime Configuration')
    regime_group.add_argument('--regime', type=str, choices=['balistic', 'batch_equivalent'], default='balistic', help='Optimization regime to use')
    regime_group.add_argument('--simulations', type=str, nargs='+', choices=['1st_order_sde', '2nd_order_sde'], default=[], help='Types of simulations to run')
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
    loss, val_loss, runs = torch.zeros(num_batched_runs, batch_size, num_steps), torch.zeros(num_batched_runs, batch_size, num_steps), torch.zeros(num_batched_runs, batch_size, num_steps, dim_result), 
    model = model_factory()

    y0_batched = y0.unsqueeze(0).expand(batch_size, -1).to(device)

    if which_approximation == 'approx_2_fun':
        dt = tau**2
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
            val_loss[run, : , t] = model.val_loss_batch(res_cont[:, t, :dim_weights])
            loss[run, : , t] = model.loss_batch(res_cont[:, t, :dim_weights])

    # Aggregate results
    loss = loss.reshape(-1, loss.shape[2])
    final_loss_distribution = loss[:, -1]
    loss = loss.mean(dim=0)
    val_loss = val_loss.reshape(-1, val_loss.shape[2]).mean(dim=0)
    runs =  runs.reshape(-1, runs.shape[2], runs.shape[3])
    if optimizer == 'Adam':
        theta, final_distribution = norm_and_mean(runs[:, :, :dim_weights])
        m, _ = norm_and_mean(runs[:, :, dim_weights: 2*dim_weights])
        v, _ = norm_and_mean(runs[:, :, 2*dim_weights:])
    elif optimizer == 'RMSProp':
        theta, final_distribution = norm_and_mean(runs[:, :, :dim_weights])
        v, _ = norm_and_mean(runs[:, :, dim_weights:])
    t1 = time.time()

    res = {
        'Loss': loss.to('cpu'),
        'Val_loss': val_loss.to('cpu'),
        'theta_mean': theta.to('cpu'),
        'v_mean': v.to('cpu'),
        'final_distribution': final_distribution.to('cpu'),
        'final_loss_distribution': final_loss_distribution.to('cpu'),
        'time_steps': ts.cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': num_batched_runs * batch_size
    }
    if optimizer == 'Adam':
        res['m_mean'] = m.to('cpu')
    
    print(f"[{which_approximation}] Simulations completed.\n")
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
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run discrete RMSProp simulations with batch processing.
    
    Returns:
        Dict of (Loss_disc, Val_loss_disc, theta_mean_disc, v_mean_disc, y0)
    """
    print("[DISCRETE] Starting discrete simulations...")
    set_seed(seed)
    t0 = time.time()
    num_batched_runs = math.ceil(num_runs / batch_size)
    
    if optimizer == 'Adam':
        dim_result = dim_weights * 3  # theta, m, v
    elif optimizer == 'RMSProp':
        dim_result = dim_weights * 2  # theta, v
    Loss_disc, Val_loss_disc, discrete_runs = torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point), torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point), torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point, dim_result)
    final_loss_distribution = torch.zeros(num_runs)
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

        # Compute validation loss with batch processing
        for t in range(num_steps - skip_initial_point):
            val_loss = model.val_loss_batch(res[:, t, :dim_weights])
            Val_loss_disc[run, : , t] = val_loss
        
    # Aggregate results
    discrete_runs = discrete_runs.reshape(-1, discrete_runs.shape[2], discrete_runs.shape[3])
    Val_loss_disc = Val_loss_disc.reshape(-1, Val_loss_disc.shape[2]).mean(dim=0)
    Loss_disc = Loss_disc.reshape(-1, Loss_disc.shape[2])
    final_loss_distribution = Loss_disc[:, -1]
    Loss_disc = Loss_disc.mean(dim=0)
    if optimizer == 'Adam':
        theta_mean_disc, final_distribution_disc = norm_and_mean(discrete_runs[:, :, :dim_weights])
        m_mean_disc, _ = norm_and_mean(discrete_runs[:, :, dim_weights: 2*dim_weights])
        v_mean_disc, _ = norm_and_mean(discrete_runs[:, :, 2*dim_weights:])
    elif optimizer == 'RMSProp':
        theta_mean_disc, final_distribution_disc = norm_and_mean(discrete_runs[:, :, :dim_weights])
        v_mean_disc, _ = norm_and_mean(discrete_runs[:, :, dim_weights:])
    t1 = time.time()

    result_disc = {
        'Loss': Loss_disc.to('cpu'),
        'Val_loss': Val_loss_disc.to('cpu'),
        'theta_mean': theta_mean_disc.to('cpu'),
        'v_mean': v_mean_disc.to('cpu'),
        'initial_point': y0.to('cpu'),
        'final_distribution': final_distribution_disc.to('cpu'),
        'final_loss_distribution': final_loss_distribution.to('cpu'),
        'time_steps': torch.arange(tau * skip_initial_point, tau * (num_steps), tau).cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': num_batched_runs * batch_size
    }
    if optimizer == 'Adam':
        result_disc['m_mean'] = m_mean_disc.to('cpu')

    print("[DISCRETE] Simulations completed.\n")
    return result_disc

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
            dim_weights, num_runs, batch_size, epsilon, sigma_value, seed, device, verbose=verbose
        )
    elif regime == 'batch_equivalent':
        return run_sde_simulations(
            model_factory, optimizer, regime_funcs, 'approx_1_fun', y0, tau, c, final_time, skip_initial_point,
            initial_params, dim_weights, num_runs, batch_size, epsilon, sigma_value, seed, device, verbose=verbose
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
    
    res_cont_1 = torchsde.sdeint(sde1, y0.unsqueeze(0), ts, method='euler', dt=tau**2).permute(1, 0, 2)
    
    # Compute losses for deterministic with batch processing
    theta_batch = res_cont_1[0, :, :dim_weights].to(device)
    loss_1_order = model.loss_batch(theta_batch)
    val_loss_1_order = model.val_loss_batch(theta_batch)
    final_loss_distribution = loss_1_order[-1]
    theta_1_order, final_distribution_1_order_det = norm_and_mean(res_cont_1[:, :, :dim_weights])
    v_1_order, _ = norm_and_mean(res_cont_1[:, :, dim_weights:2*dim_weights])
    if optimizer == 'Adam':
        m_1_order, _ = norm_and_mean(res_cont_1[:, :, 2*dim_weights:])
    t1 = time.time()

    res_1_order_det = {
        'Loss': loss_1_order.to('cpu'),
        'Val_loss': val_loss_1_order.to('cpu'),
        'theta_mean': theta_1_order.to('cpu').squeeze(1),
        'v_mean': v_1_order.to('cpu').squeeze(1),
        'final_distribution': final_distribution_1_order_det.to('cpu'),
        'final_loss_distribution': final_loss_distribution.to('cpu'),
        'time_steps': ts.cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': 1
    }
    if optimizer == 'Adam':
        res_1_order_det['m_mean'] = m_1_order.to('cpu').squeeze(1)
    print("[1ST ORDER SDE] Deterministic simulation completed.")

    # Stochastic simulations
    res_1_order_stoc = run_sde_simulations(
            model_factory, optimizer, regime_funcs, 'approx_1_fun', y0, tau, c, final_time, skip_initial_point,
            initial_params, dim_weights, num_runs, batch_size, epsilon, sigma_value, seed, device, verbose=verbose
        )
    
    print("[1ST ORDER SDE] Stochastic simulations completed.\n")
    
    return res_1_order_stoc, res_1_order_det

def run_experiment_configuration(
    args: argparse.Namespace,
    tau: float,
    sigma_value: float,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    y_train: torch.Tensor,
    y_val: torch.Tensor
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
        args.epsilon = args.epsilon / tau  

    if args.optimizer == 'Adam':
        args.c = (args.c_1, args.c_2)
        beta = (1 - tau * args.c_1, 1 - tau * args.c_2)
        print(f"\n==================== tau = {tau}, C1 = {args.c_1}, C2 = {args.c_2}, BETA = {beta}, SIGMA = {sigma_value} ====================\n")
    elif args.optimizer == 'RMSProp':
        beta = 1 - tau * args.c
        print(f"\n==================== tau = {tau}, C = {args.c}, BETA = {beta}, SIGMA = {sigma_value} ====================\n")

    # Create result directory
    result_dir = f"{args.results_dir}_tau_{tau}_c_{args.c}_sigma_{sigma_value}_finaltime_{args.final_time}"
    result_dir = os.path.join(result_dir, args.regime.replace(' ', '_'))
    os.makedirs(result_dir, exist_ok=True)
    
    # Setup model factory
    dataset = type('Dataset', (object,), {'X': X_train, 'y': y_train, 'X_val': X_val, 'y_val': y_val})()
    input_dim = args.input_dim or X_train.shape[1]
    
    def model_factory():
        return ShallowNN(input_dim, args.mid_dim, args.output_dim, dataset, device=args.device)    
    
    # Get initial parameters
    set_seed(args.seed_parameters)
    initial_model = ShallowNN(input_dim, args.mid_dim, args.output_dim, dataset, args.device)
    initial_params = initial_model.initial_weights
    dim_weights = initial_params.shape[0]
    
    # Setup time parameters
    num_steps = int(torch.ceil(torch.tensor(args.final_time / tau)).item())
    print(f'Number of steps: {num_steps}')
    
    # Get regime functions
    regime_funcs = get_regime_functions(args.regime, args.optimizer)
    
    # Generate noise
    set_seed(args.seed_noise)
    noise = sigma_value * torch.randn((5000, initial_params.shape[0]))
    
    t0 = time.time()
    
    # Run discrete simulations
    res_disc = run_discrete_simulations(
        model_factory, args.optimizer, regime_funcs, noise, tau, beta, args.c, num_steps, 
        initial_params, args.skip_initial_point, args.epsilon, args.num_runs, args.batch_size,
        dim_weights, args.seed_disc, args.device, args.verbose
    )
    y0 = res_disc['initial_point'].to(args.device)
    # Run 1st order SDE simulations
    res_1_order_det = None
    if '1st_order_sde' in args.simulations:
        res_1_order_stoc, res_1_order_det = run_1st_order_sde_simulations(
            args.regime, args.optimizer, model_factory, regime_funcs, y0, tau, args.c, args.final_time,
            args.skip_initial_point, initial_params, dim_weights,
            args.num_runs, args.batch_size, args.epsilon, sigma_value, args.seed_1st, args.device, args.verbose
        )

    # # Run 2nd order SDE simulations
    if '2nd_order_sde' in args.simulations:
        res_2_order = run_sde_simulations(
            model_factory, args.optimizer, regime_funcs, 'approx_2_fun', y0, tau, args.c, args.final_time,
            args.skip_initial_point, initial_params, dim_weights,
            args.num_runs, args.batch_size, args.epsilon, sigma_value, args.seed_2nd, args.device, args.verbose
        )
    
    t1 = time.time()
    final_results = {
        'disc': res_disc,
        'final_time': args.final_time,
        'tau': tau,
        'c': args.c,
        'sigma': sigma_value,
        'epsilon': args.epsilon,
        'regime': args.regime,
        'optimizer': args.optimizer,
        'total_time_elapsed': t1 - t0,
        'skipped_initial_points': args.skip_initial_point,
        'simulation keys': ['disc']
    }
    if '1st_order_sde' in args.simulations:
        final_results['1_order_stoc'] = res_1_order_stoc
        final_results['simulation keys'] += ['1_order_stoc']
    if '2nd_order_sde' in args.simulations:
        final_results['2_order_stoc'] = res_2_order
        final_results['simulation keys'] += ['2_order_stoc']
    if res_1_order_det is not None:
        final_results['1_order_det'] = res_1_order_det
        final_results['simulation keys'] += ['1_order_det']

    # Save results
    torch.save(final_results, os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}_sigma{sigma_value}.pt'))
   
    effective_runs = math.ceil(args.num_runs / args.batch_size) * args.batch_size

    # wandb logging
    if args.wandb:
        for sim in final_results['simulation keys']:
            wandb.init(
                project='LongTime-ShallowNN-CaliforniaHousing-RMSProp_',
                name=f'{args.optimizer}_{regime_name}_{sim}_tau_{tau}_c_{args.c}_sigma_{sigma_value}_nruns_{effective_runs}',
                config=vars(args),
                notes='Comparison of discrete RMSProp with SDE approximations for shallow NN on California Housing dataset with comparison of loss, validation loss, norm of the theta and v and distribution of the final loss and final theta.',
                save_code=True
            )
            artifact = wandb.Artifact(f"final_results_tau_{tau}_sigma_{sigma_value}", type="results")
            artifact.add_file(os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}_sigma{sigma_value}.pt'))
            wandb.log_artifact(artifact)
            wandb.log({"time_elapsed": final_results[sim]['time_elapsed'], 'runs': effective_runs})

            ts = final_results[sim]['time_steps'].numpy()
            for t in range(len(ts)):
                loss_val = final_results[sim]['Loss'][t].item()
                val_loss_val = final_results[sim]['Val_loss'][t].item()
                theta_mean = final_results[sim]['theta_mean'][t].item()
                v_mean = final_results[sim]['v_mean'][t].item()
                
                wandb.log({
                    f"Loss": loss_val,
                    f"Val_loss": val_loss_val,
                    f"theta": theta_mean,
                    f"v": v_mean,
                    "time": ts[t]
                })
      

            # Log final distributions as histograms to wandb
            fd = final_results[sim]['final_distribution'].cpu().numpy().flatten()
            table = wandb.Table(data=[[v] for v in fd], columns=["value"])
            wandb.log({
                f"Histogram/final_theta_distribution":
                    wandb.plot.histogram(table, "value", title=f"{final_results['regime']} {sim} Final Distribution")
            })


            fld = final_results[sim]['final_loss_distribution'].cpu().numpy()
            if fld.ndim == 0:
                fld = fld.reshape(-1)
            table = wandb.Table(data=[[v] for v in fld], columns=["value"])
            wandb.log({
                f"Histogram/final_loss_distribution":
                    wandb.plot.histogram(table, "value", title=f"{final_results['regime']} {sim} Final Loss Distribution")
            })

            wandb.finish()
    print(f"[tau={tau}] Execution time: {t1-t0:.2f} seconds\n")
    

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    print("Starting Neural Network Training with RMSProp SDE Approximations")
    print(f"Regime: {args.regime}")
    print(f"Parameters: tau={args.tau_list}, c={args.c}, sigma={args.sigma_list}")
    print(f"Number of runs: {args.num_runs}")
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(args.test_size, args.random_state)
    
    
    # Run experiments for all parameter combinations
    for tau in args.tau_list:
        for sigma_value in args.sigma_list:
            run_experiment_configuration(
                args, tau, sigma_value, X_train, X_val, y_train, y_val
            )
    
    print("All experiments completed successfully!")



if __name__ == "__main__":
    main()


'''
python -m NeuralNetwork.main_v3 --regime balistic --optimizer RMSProp; python -m NeuralNetwork.main_v3 --regime balistic --optimizer Adam; python -m NeuralNetwork.main_v3 --regime batch_equivalent --optimizer Adam; python -m NeuralNetwork.main_v3 --regime batch_equivalent --optimizer RMSProp;
'''