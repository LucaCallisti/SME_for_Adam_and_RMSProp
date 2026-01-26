import os
import random
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from Poly.Plot_poly import plot_poly_result

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    Args:
        seed: Random seed value.
    Returns:
        None
    """
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


def processing_outputs(tensor: torch.Tensor, checkpoints: torch.Tensor, variable : str) -> torch.Tensor:
    """
    Compute mean, std, and distributions at checkpoints for a tensor.
    Args:
        tensor: Input tensor.
        checkpoints: Tensor of checkpoint locations (0-1).
        variable: Name of the variable being processed.
    Returns:
        Dictionary with mean, std, and distributions at checkpoints.
    """
    mean = tensor.mean(dim=0)
    std_dev = torch.std(tensor, dim=0)
    if torch.isnan(std_dev).all():
        std_dev = torch.zeros_like(mean)
    mean_down = mean - std_dev
    mean_up = mean + std_dev
    final_result = torch.stack([mean_down.unsqueeze(0), mean.unsqueeze(0), mean_up.unsqueeze(0)], dim=0)

    shapes = tensor.shape[1]
    checkpoints_index = (checkpoints * (shapes - 1)).long()
    distributions_norm = tensor[:, checkpoints_index.squeeze(), :]

    result = {
        variable: final_result.squeeze(),
        variable + '_distributions': distributions_norm,
        variable + '_mean': mean,
        variable + '_std_dev': std_dev,
        'checkpoints': checkpoints
    }
    return result

def processing_outputs_caller(runs, dim_weights, checkpoints, optimizer):
    """
    Process simulation outputs for theta, m, v depending on optimizer.
    Args:
        runs: Simulation output tensor.
        dim_weights: Number of weights.
        checkpoints: Checkpoint tensor.
        optimizer: Optimizer name.
    Returns:
        Dictionary with processed outputs.
    """
    if optimizer == 'Adam':
        res_theta = processing_outputs(runs[:, :, :dim_weights], checkpoints=checkpoints, variable='theta')
        res_m = processing_outputs(runs[:, :, dim_weights: 2*dim_weights], checkpoints=checkpoints, variable='m')
        res_v = processing_outputs(runs[:, :, 2*dim_weights:], checkpoints=checkpoints, variable='v')
        res = {
            **res_theta,
            **res_m,
            **res_v
        }
    elif optimizer == 'RMSProp':
        res_theta = processing_outputs(runs[:, :, :dim_weights], checkpoints=checkpoints, variable='theta')
        res_v = processing_outputs(runs[:, :, dim_weights:], checkpoints=checkpoints, variable='v')
        res = {
            **res_theta,
            **res_v
        }
    return res

def log_to_wandb(final_results, args, initial_points_before_disc, sigma_value, result_dir, tau, poly, effective_runs, final_time, plot_only_images = False):
    """
    Log experiment results to Weights & Biases (wandb) and save results to disk.
    Args:
        final_results: Dictionary of results.
        args: Argument namespace.
        initial_points_before_disc: Initial parameter tensor.
        sigma_value: Noise scale.
        result_dir: Directory to save results.
        tau: Step size.
        poly: Polynomial function object.
        effective_runs: Number of effective runs.
        final_time: Final simulation time.
        plot_only_images: If True, only log images.
    Returns:
        None
    """

    config = vars(args).copy()
    config.update({'initial point bf disc' : initial_points_before_disc.item()})

    wandb.init(
        project=args.name_project,
        entity='Effective-continuous-equations',
        name=f'Noise{args.noise_level}_{args.optimizer}{args.regime}_{initial_points_before_disc.item():.2f}_sigma{sigma_value:.2f}_BatchSize{args.batch_size_simulation}_tau{tau}_c{args.c}_time{final_time}',
        config=config,
        notes='Comparison of discrete RMSProp with SDE approximations for shallow NN on California Housing dataset with comparison of loss, validation loss, norm of the theta and v and distribution of the final loss and final theta.',
        save_code=True
    )

    save_path = os.path.join(result_dir, f'final_results_{args.regime}_tau{tau}_sigma{sigma_value}.pt')
    torch.save(final_results, save_path)
    artifact = wandb.Artifact(f"final_results_tau_{tau}_sigma_{sigma_value}", type="results")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    
    def aux_plot_mean_and_std(title: str, final_results) -> None:
        sims = final_results['simulation keys']
        ts = final_results[sims[0]]['time_steps'].numpy()

        plt.figure()
        colors = ['blue', 'orange', 'green', 'red']
        linestyles = ['-', '--', '-.', ':']
        labels = {
            'disc': 'Discrete',
            '1_order_stoc': '1st-Order',
            '2_order_stoc': '2nd-Order',
            'loss': 'Loss'
        }

        for i, sim in enumerate(sims):
            ts = final_results[sim]['time_steps'].numpy()
            down = final_results[sim][title][0]
            mean = final_results[sim][title][1]
            up = final_results[sim][title][2]

            if ts.shape[0] != down.shape[0]:
                ts = ts[:down.shape[0]]

            plt.fill_between(ts, down.cpu().numpy(), up.cpu().numpy(), alpha=0.3)
            if sim in labels:
                lab = labels[sim]
            else:
                lab = sim
            plt.plot(ts, mean.cpu().numpy(), label=lab, color=colors[i], linestyle=linestyles[i])

        plt.title(title.capitalize() + ' over Time with $\\alpha$ = ' + str(args.noise_level))
        plt.xlabel('Time')
        plt.grid()
        plt.legend()
        wandb.log({f"{title}_plot":  wandb.Image(plt)})
        plt.close()

    aux_plot_mean_and_std('theta', final_results)
    aux_plot_mean_and_std('v', final_results)
    if args.optimizer == 'Adam':
        aux_plot_mean_and_std('m', final_results)
    aux_plot_mean_and_std('loss', final_results)

    for sim in final_results['simulation keys']:
        artifact = wandb.Artifact(f"final_results_tau_{tau}_sigma_{sigma_value}", type="results")
        artifact.add_file(os.path.join(result_dir, f'results_regime{args.regime}_tau{tau}_c{args.c}.pt'))
        wandb.log_artifact(artifact)
        wandb.log({f"Barplot/time_elapsed_for_{sim}": final_results[sim]['time_elapsed'], 'runs': effective_runs})
        ts = final_results[sim]['time_steps'].numpy()
        theta_down = final_results[sim]['theta'][0]
        theta_mean = final_results[sim]['theta'][1]
        theta_up = final_results[sim]['theta'][2]
        v_down = final_results[sim]['v'][0]
        v_mean = final_results[sim]['v'][1]
        v_up = final_results[sim]['v'][2]
        loss_down = final_results[sim]['loss'][0]
        loss_mean = final_results[sim]['loss'][1]
        loss_up = final_results[sim]['loss'][2]
        if args.optimizer == 'Adam':
            m_down = final_results[sim]['m'][0]
            m_mean = final_results[sim]['m'][1]
            m_up = final_results[sim]['m'][2]
        if plot_only_images == False:
            max_index = min(theta_down.shape[0], ts.shape[0])
            for t in range(max_index):
                t_down, t_mean, t_up = theta_down[t].item(), theta_mean[t].item(), theta_up[t].item()
                v_down_t, v_mean_t, v_up_t = v_down[t].item(), v_mean[t].item(), v_up[t].item()
                loss_down_t, loss_mean_t, loss_up_t = loss_down[t].item(), loss_mean[t].item(), loss_up[t].item()
                dict_to_log = {
                    f"theta_down_{sim}": t_down,
                    f"theta_{sim}": t_mean,
                    f"theta_up_{sim}": t_up,
                    f"v_down_{sim}": v_down_t,
                    f"v_{sim}": v_mean_t,
                    f"v_up_{sim}": v_up_t,
                    f"loss_down_{sim}": loss_down_t,
                    f"loss_{sim}": loss_mean_t,
                    f"loss_up_{sim}": loss_up_t,
                    "time": ts[t]
                }                
                if args.optimizer == 'Adam':
                    m_down_t, m_mean_t, m_up_t = m_down[t].item(), m_mean[t].item(), m_up[t].item()

                    dict_to_log.update({
                        f"m_down_{sim}": m_down_t,
                        f"m_{sim}": m_mean_t,
                        f"m_up_{sim}": m_up_t
                    })
                wandb.log(dict_to_log)
        
        def _aux_log_distribution(name: str, data: torch.Tensor, title: str, t : float) -> None:
            data_np = data.cpu().numpy().flatten()
            table = wandb.Table(data=[[v] for v in data_np], columns=["value"])
            res = {
                f"Histogram_{name}/time_{t:.5f}":
                    wandb.plot.histogram(table, "value", title=title+f" at time {t:.5f}"),
            }
            return res
            

        keys = [k if 'distribution' in k else None for k in final_results[sim].keys()]
        res_to_log = {}
        for key in keys:
            if key is not None:
                ts = final_results[sim]['time_steps']
                checkpoints = final_results[sim]['checkpoints']
                index_chekpoints = (checkpoints * (ts.shape[0]-1)).long()
                for i, index in enumerate(index_chekpoints):
                    res_to_log.update(_aux_log_distribution(
                        name=key,
                        data=final_results[sim][key][:, i],
                        title=key,
                        t=ts[index]
                    ))
        wandb.log(res_to_log)
    plot_poly_result(final_results, poly, tau, result_dir, args)

    wandb.finish()

def calculate_loss(poly, theta):
    """
    Calculate the loss for a batch of theta values using the polynomial function.
    Args:
        poly: Polynomial function object.
        theta: Batch of parameter values.
    Returns:
        Tensor with mean, std, and bounds of the loss.
    """
    with torch.no_grad():
        loss = poly.f(theta)
    if loss.ndim == 3:
        loss = loss.squeeze(-1)
    mean = loss.mean(dim=0)
    std_dev = torch.std(loss, dim=0)
    loss_mean = mean
    loss_mean_up = mean + std_dev
    loss_mean_down = mean - std_dev
    loss = torch.stack([loss_mean_down.unsqueeze(0), loss_mean.unsqueeze(0), loss_mean_up.unsqueeze(0)], dim=0)
    if loss.ndim == 3:
        loss = loss.squeeze()
    return loss