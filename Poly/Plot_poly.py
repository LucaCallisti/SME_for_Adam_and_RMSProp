import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb

def plot_poly_result(final_results, poly, tau, result_dir, args = None, only_last = False, to_plot = ['disc', '1_order_stoc', '2_order_stoc']):
    initial_before = final_results['initial_points_before_disc'].cpu().numpy()
    initial_after = final_results['initial_points_after_disc'][0].cpu().numpy()


    liminf = poly.x_liminf
    limsup = poly.x_limsup
    if liminf > initial_before:
        liminf = float(initial_before)
    if limsup < initial_before:
        limsup = float(initial_before)
    x = torch.linspace(liminf, limsup, 1000)

    vertical_scaling_factor = - 1.02

    f1 = 0.5 * poly.f1(x).cpu().numpy()
    f2 = 0.5 * poly.f2(x).cpu().numpy()
    f = poly.f(x).cpu().numpy() + vertical_scaling_factor
    x_np = x.cpu().numpy()

    f_initial_before = poly.f(torch.as_tensor(initial_before)).cpu().numpy() + vertical_scaling_factor
    f_initial_after = poly.f(torch.as_tensor(initial_after)).cpu().numpy() + vertical_scaling_factor

    to_plot = [x for x in to_plot if x in final_results.keys()]

    number_of_checkpoints = final_results[to_plot[0]]['theta_distributions'].shape[1]
    final_time = final_results['final_time']
    res_to_log = {}
    if only_last == False:
        start_index = 1
    else:
        start_index = number_of_checkpoints - 1
    for i in range(start_index, number_of_checkpoints):
        dist_to_plot = []
        for sim in to_plot:
            dist_to_plot.append(final_results[sim]['theta_distributions'][:, i].cpu().numpy().flatten())

        fig, ax1 = plt.subplots(figsize=(10,6))

        ax1.plot(x_np, f, label='f(x)', color='black', linewidth=2, zorder = 1)
        ax1.plot(x_np, f1, label='f1(x)', color='blue', linestyle='--', zorder = 1)
        ax1.plot(x_np, f2, label='f2(x)', color='green', linestyle='--', zorder = 1)

        ax1.scatter(initial_before, f_initial_before, color='red', marker='o', label='initial points before disc', zorder = 2)
        ax1.scatter(initial_after, f_initial_after, color='orange', marker='x', label='initial points after disc', zorder = 2)

        ax1.set_xlabel('x')
        ax1.set_ylabel('Function values')

        ax1.set_xlim(-1.3, 1.6)
        ax1.set_ylim(0, 2)

        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()

        bins = 200
        colors = ['#000080', '#FF8000', '#808000', '#800080', '#008080']
        label = {'disc': 'Final dist (disc)', '1_order_stoc': 'Final dist 1st', '2_order_stoc': 'Final dist 2nd', '2_order_Balistic': 'Final dist 2nd Balistic', '2_order_BatchEq': 'Final dist 2nd BatchEq'}
        for i, dist in enumerate(dist_to_plot):
            ax2.hist(dist, bins=bins, alpha=0.3, color=colors[i], label=label[to_plot[i]], weights=np.ones_like(dist), density=True)

        ax2.set_ylim(0, 15)
        ax2.set_ylabel('Probability density')
        ax2.legend(loc='upper right')

        fig_path = os.path.join(result_dir, f'final_plot_tau_{tau}.png')
        fig.savefig(fig_path)
        time = final_time * (i / (number_of_checkpoints-1))
        res_to_log[f"Distributions/Distribution_at_{time}"] =  wandb.Image(fig_path)
        plt.close(fig)

    if args is not None:
        if args.wandb:
            wandb.log(res_to_log)
    else:
        wandb.log(res_to_log)