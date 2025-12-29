import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb

def plot_poly_result(final_results, poly, tau, result_dir, args):
    initial_before = final_results['initial_points_before_disc'].cpu().numpy()
    initial_after = final_results['initial_points_after_disc'][0].cpu().numpy()


    liminf = poly.x_liminf
    limsup = poly.x_limsup
    if liminf > initial_before:
        liminf = float(initial_before)
    if limsup < initial_before:
        limsup = float(initial_before)
    x = torch.linspace(liminf, limsup, 1000)

    vertical_scaling_factor = - 1.0

    f1 = 0.5 * poly.f1(x).cpu().numpy()
    f2 = 0.5 * poly.f2(x).cpu().numpy()
    f = poly.f(x).cpu().numpy() + vertical_scaling_factor
    x_np = x.cpu().numpy()

    f_initial_before = poly.f(torch.as_tensor(initial_before)).cpu().numpy() + vertical_scaling_factor
    f_initial_after = poly.f(torch.as_tensor(initial_after)).cpu().numpy() + vertical_scaling_factor

    dist_disc = final_results['disc']['final_distribution'].cpu().numpy().flatten()
    dist_1st = final_results['1_order_stoc']['final_distribution'].cpu().numpy().flatten() if '1_order_stoc' in final_results else None
    dist_2nd = final_results['2_order_stoc']['final_distribution'].cpu().numpy().flatten() if '2_order_stoc' in final_results else None

    print(f'mean of distribution disc: {dist_disc.mean()},1st order: {dist_1st.mean() if dist_1st is not None else "N/A"}, 2nd order: {dist_2nd.mean() if dist_2nd is not None else "N/A"}')

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Plot delle funzioni
    ax1.plot(x_np, f, label='f(x)', color='black', linewidth=2, zorder = 1)
    ax1.plot(x_np, f1, label='f1(x)', color='blue', linestyle='--', zorder = 1)
    ax1.plot(x_np, f2, label='f2(x)', color='green', linestyle='--', zorder = 1)

    # Plot dei punti iniziali
    ax1.scatter(initial_before, f_initial_before, color='red', marker='o', label='initial points before disc', zorder = 2)
    ax1.scatter(initial_after, f_initial_after, color='orange', marker='x', label='initial points after disc', zorder = 2)

    ax1.set_xlabel('x')
    ax1.set_ylabel('Function values')

    ax1.set_xlim(-1.3, 1.6)
    ax1.set_ylim(0, 2)

    ax1.legend(loc='upper left')

    # Istogramma delle distribuzioni finali (su un secondo asse y)
    ax2 = ax1.twinx()

    bins = 200
    # Moltiplichiamo gli istogrammi per il fattore di scala
    ax2.hist(dist_disc, bins=bins, alpha=0.3, color='#000080', label='Final dist (disc)', weights=np.ones_like(dist_disc), density=True)
    if dist_1st is not None:
        ax2.hist(dist_1st, bins=bins, alpha=0.3, color='#FF8000', label='Final dist 1st', weights=np.ones_like(dist_1st), density=True)
    if dist_2nd is not None:
        ax2.hist(dist_2nd, bins=bins, alpha=0.3, color='#808000', label='Final dist 2nd', weights=np.ones_like(dist_2nd), density=True)

    ax2.set_ylim(0, 15)
    ax2.set_ylabel('Probability density')
    ax2.legend(loc='upper right')

    fig_path = os.path.join(result_dir, f'final_plot_tau_{tau}.png')
    print(f'Saving final plot to {fig_path}')
    fig.savefig(fig_path)
    if args.wandb:
        wandb.log({"final_plot": wandb.Image(fig_path)})
    plt.close(fig)