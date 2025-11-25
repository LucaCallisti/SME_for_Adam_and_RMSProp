"""
This script compares discrete RMSProp optimization with various SDE (Stochastic Differential Equation) 
approximations for function approximation. It supports both ballistic and batch equivalent regimes.
"""

import argparse
import os
import time
from typing import Dict, Tuple, Optional, Any
import sys

import torch
import torchsde
import wandb
import math


from Poly.Utils import set_seed, processing_outputs
from Poly.poly import function_poly 
from Algorithms.Utils import get_regime_functions
from Poly.Plot_poly import plot_poly_result

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.setrecursionlimit(10000)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive parameter configuration."""
    parser = argparse.ArgumentParser(
        description='Neural Network Training with RMSProp SDE Approximations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Function parameters
    func_group = parser.add_argument_group('Function Configuration')
    func_group.add_argument('--points', type=float, nargs='+', default=[-1.0, 1.0, 0.25, 2], help='(x, y) points for Hermite Quintic Polynomial: x1 y1 x2 y2 xM yM')

    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    # 1.5, 1.15, 0.85, 0.5, 0.1, 0.0, -0.1,
    train_group.add_argument('--initial_points', type=float, default=[1.5, 1.15, 0.85, 0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5], help='Initial points for optimization')
    train_group.add_argument('--tau-list', type=float, nargs='+', default=[0.01], help='Learning rate values to test')
    train_group.add_argument('--c', type=float, default=0.5, help='RMSProp scaling constant of beta')
    train_group.add_argument('--c-1', type=float, default=1, help='C 1 parameter for Adam optimizer')
    train_group.add_argument('--c-2', type=float, default=0.5, help='C 2 parameter for Adam optimizer')
    train_group.add_argument('--sigma-list', type=float, nargs='+', default=[0.2], help='Noise variance values to test')
    train_group.add_argument('--num-runs', type=int, default=1024, help='Number of simulation runs for averaging')
    train_group.add_argument('--final-time', type=float, default=15.0, help='Final time for SDE integration')
    train_group.add_argument('--epsilon', type=float, default=0.1, help='Regularization epsilon for RMSProp')
    train_group.add_argument('--skip-initial-point', type=int, default=2, help='Number of initial points to skip in analysis')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run simulations on (cpu or cuda)')
    train_group.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')

    # Regime selection
    regime_group = parser.add_argument_group('Regime Configuration')
    regime_group.add_argument('--regime', type=str, choices=['balistic', 'batch_equivalent'], default='batch_equivalent', help='Optimization regime to use')
    regime_group.add_argument('--simulations', type=str, nargs='+', choices=['1st_order_sde', '2nd_order_sde'], default=['1st_order_sde', '2nd_order_sde'], help='Types of simulations to run')
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
    
    return parser.parse_args()

def run_sde_simulations(
        poly: function_poly,
        optimizer: str,
        regime_funcs: Dict[str, Any],
        which_approximation: str,
        initial_points: torch.Tensor,
        tau: float,
        c: float,
        final_time: float,
        skip_initial_point: int,
        dim_weights: int,
        num_runs: int,
        batch_size: int,
        epsilon: float,
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
    runs = torch.zeros(num_batched_runs, batch_size, num_steps, dim_result) 

    if which_approximation == 'approx_2_fun':
        dt = tau**2
    elif which_approximation == 'approx_1_fun':
        dt = tau

    for run in range(num_batched_runs):
        if verbose:
            print(f"[{which_approximation}] Simulation {run+1}/{num_batched_runs}...")
        
        # Setup regularizer
        regime_funcs['regularizer'].set_costant(torch.tensor(tau))
        
        # Create and run SDE
        sde = regime_funcs[which_approximation](
            tau, c, poly, ts, regime_funcs['regularizer'], 
            Verbose=True, epsilon=epsilon, constant_noise = False
        )

        res_cont = torchsde.sdeint(sde, initial_points.unsqueeze(0).expand(batch_size, -1).to(device), ts, method='euler', dt=dt).permute(1, 0, 2)
        runs[run] = res_cont


    # Aggregate results
    runs =  runs.reshape(-1, runs.shape[2], runs.shape[3])
    if optimizer == 'Adam':
        theta, final_distribution = processing_outputs(runs[:, :, :dim_weights])
        m, _ = processing_outputs(runs[:, :, dim_weights: 2*dim_weights])
        v, _ = processing_outputs(runs[:, :, 2*dim_weights:])
    elif optimizer == 'RMSProp':
        theta, final_distribution = processing_outputs(runs[:, :, :dim_weights])
        v, _ = processing_outputs(runs[:, :, dim_weights:])
    t1 = time.time()

    res = {
        'theta_mean': theta.to('cpu'),
        'v_mean': v.to('cpu'),
        'final_distribution': final_distribution.to('cpu'),
        'time_steps': ts.cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': num_batched_runs * batch_size
    }
    if optimizer == 'Adam':
        res['m_mean'] = m.to('cpu')
    
    print(f"[{which_approximation}] Simulations completed.\n")
    return res

def run_discrete_simulations(
    poly: function_poly,
    optimizer: str,
    regime_funcs: Dict[str, Any],
    noise: torch.Tensor,
    tau: float,
    beta: float,
    c: float,
    num_steps: int,
    initial_points: torch.Tensor,
    skip_initial_point: int,
    epsilon: float,
    num_runs: int,
    batch_size: int,
    dim_weights: int,
    seed: int,
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
    discrete_runs = torch.zeros(num_batched_runs, batch_size, num_steps - skip_initial_point, dim_result)
    final_loss_distribution = torch.zeros(num_runs)
    y0 = None
    
    for run in range(num_batched_runs):
        if verbose:
            print(f"[DISCRETE] Simulation {run+1}/{num_batched_runs}...")

        res = regime_funcs['discr_fun'](
            poly, noise, tau, beta, c, num_steps, initial_points.unsqueeze(0).expand(batch_size, -1),
            skip_initial_point, epsilon=epsilon, loss_bool = False
        )

        discrete_runs[run] = res[:, skip_initial_point:, :]
        
        if y0 is None:
            y0 = res[0, skip_initial_point, :]
        
    # Aggregate results
    discrete_runs = discrete_runs.reshape(-1, discrete_runs.shape[2], discrete_runs.shape[3])
    if optimizer == 'Adam':
        theta_mean_disc, final_distribution_disc = processing_outputs(discrete_runs[:, :, :dim_weights])
        m_mean_disc, _ = processing_outputs(discrete_runs[:, :, dim_weights: 2*dim_weights])
        v_mean_disc, _ = processing_outputs(discrete_runs[:, :, 2*dim_weights:])
    elif optimizer == 'RMSProp':
        theta_mean_disc, final_distribution_disc = processing_outputs(discrete_runs[:, :, :dim_weights])
        v_mean_disc, _ = processing_outputs(discrete_runs[:, :, dim_weights:])
    t1 = time.time()

    result_disc = {
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
    poly: function_poly,
    regime_funcs: Dict[str, Any],
    initial_points: torch.Tensor,
    tau: float,
    c: float,
    final_time: float,
    skip_initial_point: int,
    dim_weights: int,
    num_runs: int,
    batch_size: int,
    epsilon: float,
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run 1st order SDE simulations (regime-dependent).
    
    Returns:
        Tuple of results, some may be None depending on regime
    """
    ts = torch.arange(tau * skip_initial_point, final_time, tau)
    
    if regime == 'balistic':
        return _run_1st_order_balistic(
            poly, optimizer, regime_funcs, initial_points, tau, c, final_time, skip_initial_point, ts,
            dim_weights, num_runs, batch_size, epsilon, seed, device, verbose=verbose
        )
    elif regime == 'batch_equivalent':
        return run_sde_simulations(
            poly, optimizer, regime_funcs, 'approx_1_fun', initial_points, tau, c, final_time, skip_initial_point,
            dim_weights, num_runs, batch_size, epsilon, seed, device, verbose=verbose
        ), None

def _run_1st_order_balistic(
    poly: function_poly,
    optimizer: str,
    regime_funcs: Dict[str, Any],
    initial_points: torch.Tensor,
    tau: float,
    c: float,
    final_time: float,
    skip_initial_point: int,
    ts: torch.Tensor,
    dim_weights: int,
    num_runs: int,
    batch_size: int,
    epsilon: float,
    seed: int,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run 1st order SDE simulations for ballistic regime with batch processing."""
    print("[1ST ORDER SDE] Starting deterministic simulation (1st order)...")
    t0 = time.time()

    # Deterministic simulation
    regime_funcs['regularizer'].set_costant(torch.tensor(tau))

    sde1 = regime_funcs['approx_1_fun_det'](
        tau, c, poly, ts, regime_funcs['regularizer'],
        Verbose=verbose, epsilon=epsilon
    )
    
    res_cont_1 = torchsde.sdeint(sde1, initial_points.unsqueeze(0).expand(batch_size, -1), ts, method='euler', dt=tau**2).permute(1, 0, 2)
    
    # Compute losses for deterministic with batch processing
    theta_1_order, final_distribution_1_order_det = processing_outputs(res_cont_1[:, :, :dim_weights])
    v_1_order, _ = processing_outputs(res_cont_1[:, :, dim_weights: 2*dim_weights])
    if optimizer == 'Adam':
        m_1_order, _ = processing_outputs(res_cont_1[:, :, 2*dim_weights:])
    t1 = time.time()

    res_1_order_det = {
        'theta_mean': theta_1_order.to('cpu').squeeze(1),
        'v_mean': v_1_order.to('cpu').squeeze(1),
        'final_distribution': final_distribution_1_order_det.to('cpu'),
        'time_steps': ts.cpu(),
        'time_elapsed': t1 - t0,
        'n_runs': 1
    }
    if optimizer == 'Adam':
        res_1_order_det['m_mean'] = m_1_order.to('cpu').squeeze(1)
    print("[1ST ORDER SDE] Deterministic simulation completed.")

    # Stochastic simulations
    res_1_order_stoc = run_sde_simulations(
            poly, optimizer, regime_funcs, 'approx_1_fun', initial_points, tau, c, final_time, skip_initial_point,
            dim_weights, num_runs, batch_size, epsilon, seed, device, verbose=verbose
        )
    
    print("[1ST ORDER SDE] Stochastic simulations completed.\n")
    
    return res_1_order_stoc, res_1_order_det

def run_experiment_configuration(
    args: argparse.Namespace,
    tau: float,
    points: list,
    initial_points_before_disc: torch.Tensor = None
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
        print(f"\n==================== tau = {tau}, C1 = {args.c_1}, C2 = {args.c_2}, BETA = {beta} Starting point {initial_points_before_disc} ====================\n")
    elif args.optimizer == 'RMSProp':
        beta = 1 - tau * args.c
        print(f"\n==================== tau = {tau}, C = {args.c}, BETA = {beta} Starting point {initial_points_before_disc} ====================\n")

    # Create result directory
    result_dir = f"{args.results_dir}_tau_{tau}_c_{args.c}_finaltime_{args.final_time}"
    result_dir = os.path.join(result_dir, args.regime.replace(' ', '_'))
    os.makedirs(result_dir, exist_ok=True)
    
    # Create function
    poly = function_poly(*points)
    
    # Setup time parameters
    num_steps = int(torch.ceil(torch.tensor(args.final_time / tau)).item())
    print(f'Number of steps: {num_steps}')
    
    # Get regime functions
    regime_funcs = get_regime_functions(args.regime, args.optimizer)
    
    # Get initial points
    set_seed(args.seed_parameters)
    if initial_points_before_disc is None:
        initial_points_before_disc = torch.rand(1)
    dim_weights = 1

    # Generate noise
    set_seed(args.seed_noise)
    noise = torch.randint(0, 2, (5000, initial_points_before_disc.shape[0]))
    
    # Setup time parameters
    num_steps = int(torch.ceil(torch.tensor(args.final_time / tau)).item())
    print(f'Number of steps: {num_steps}')
        
    t0 = time.time()
    
    # Run discrete simulations
    res_disc = run_discrete_simulations(
        poly, args.optimizer, regime_funcs, noise, tau, beta, args.c, num_steps, 
        initial_points_before_disc, args.skip_initial_point, args.epsilon, args.num_runs, args.batch_size,
        dim_weights, args.seed_disc, args.verbose
    )
    initial_points = res_disc['initial_point'].to(args.device)
    # Run 1st order SDE simulations
    res_1_order_det = None
    if '1st_order_sde' in args.simulations:
        res_1_order_stoc, res_1_order_det = run_1st_order_sde_simulations(
            args.regime, args.optimizer, poly, regime_funcs, initial_points, tau, args.c, args.final_time,
            args.skip_initial_point, dim_weights,
            args.num_runs, args.batch_size, args.epsilon, args.seed_1st, args.device, args.verbose
        )

    # Run 2nd order SDE simulations
    if '2nd_order_sde' in args.simulations:
        res_2_order = run_sde_simulations(
            poly, args.optimizer, regime_funcs, 'approx_2_fun', initial_points, tau, args.c, args.final_time,
            args.skip_initial_point, dim_weights,
            args.num_runs, args.batch_size, args.epsilon, args.seed_2nd, args.device, args.verbose
        )
    
    t1 = time.time()
    final_results = {
        'disc': res_disc,
        'final_time': args.final_time,
        'tau': tau,
        'c': args.c,
        'epsilon': args.epsilon,
        'regime': args.regime,
        'optimizer': args.optimizer,
        'total_time_elapsed': t1 - t0,
        'skipped_initial_points': args.skip_initial_point,
        'initial_points_before_disc': initial_points_before_disc,
        'initial_points_after_disc': initial_points,
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
    torch.save(final_results, os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}.pt'))
   
    effective_runs = math.ceil(args.num_runs / args.batch_size) * args.batch_size

    # wandb logging
    if args.wandb:
        wandb.init(
            project='Poly',
            name=f'{args.optimizer}{args.regime}_{initial_points_before_disc.item():.2f}_tau{tau}_c{args.c}_time{args.final_time}',
            config=vars(args),
            notes='Comparison of discrete RMSProp with SDE approximations for shallow NN on California Housing dataset with comparison of loss, validation loss, norm of the theta and v and distribution of the final loss and final theta.',
            save_code=True
        )
        for sim in final_results['simulation keys']:
            artifact = wandb.Artifact(f"final_results_tau_{tau}", type="results")
            artifact.add_file(os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}.pt'))
            wandb.log_artifact(artifact)
            wandb.log({f"time_elapsed_for_{sim}": final_results[sim]['time_elapsed'], 'runs': effective_runs})

            ts = final_results[sim]['time_steps'].numpy()
            for t in range(len(ts)):
                theta_val = final_results[sim]['theta_mean'][t].item()
                v_val = final_results[sim]['v_mean'][t].item()
                
                wandb.log({
                    f"theta_{sim}": theta_val,
                    f"v_{sim}": v_val,
                    "time": ts[t]
                })

            final_distribution = final_results[sim]['final_distribution'].cpu().numpy().flatten()
            positive = (final_distribution >= 0).sum()
            negative = (final_distribution < 0).sum()
            wandb.log({
                f"final_theta_positive_fraction_{sim}": positive / final_distribution.shape[0],
                f"final_theta_negative_fraction_{sim}": negative / final_distribution.shape[0]
            })
      

            # Log final distributions as histograms to wandb
            fd = final_results[sim]['final_distribution'].cpu().numpy().flatten()
            table = wandb.Table(data=[[v] for v in fd], columns=["value"])
            wandb.log({
                f"Histogram/final_theta_distribution_{sim}":
                    wandb.plot.histogram(table, "value", title=f"{final_results['regime']} {sim} Final Distribution")
            })

        plot_poly_result(final_results, poly, tau, result_dir, args)
        if final_results['disc']['final_distribution'].mean() > 0:
            plot_poly_result(final_results, poly, tau, result_dir, args, xlim = (0.5, 1.5))
        else:
            plot_poly_result(final_results, poly, tau, result_dir, args, xlim = (-1.5, -0.5))

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
    
    
    # Run experiments for all parameter combinations
    for tau in args.tau_list:
        for initial_point in args.initial_points:
            initial_points_before_disc = torch.tensor([initial_point])
            run_experiment_configuration(
                args, tau, args.points, initial_points_before_disc
            )
    
    print("All experiments completed successfully!")



if __name__ == "__main__":
    main()

"""
python -m Poly.main --regime balistic --optimizer RMSProp; python -m Poly.main --regime balistic --optimizer Adam; python -m Poly.main --regime batch_equivalent --optimizer Adam; python -m Poly.main --regime batch_equivalent --optimizer RMSProp;
"""