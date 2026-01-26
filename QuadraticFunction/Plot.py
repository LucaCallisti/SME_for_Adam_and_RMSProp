import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import numpy as np



def plot_v_and_theta_single_eta(Run_d, Run_c_1, Run_c_2, eta, final_time, i, folder_path = None, power = '10', C_theta_1ord=1, C_theta_2ord=1, C_m_1ord=1, C_m_2ord=1, C_v_1ord=1, C_v_2ord=1, regime = 'Balistic'):
    """
    Plots the error and dynamics for theta, m, and v for a single value of eta.
    Args:
        Run_d: Discrete run results.
        Run_c_1: First continuous run results.
        Run_c_2: Second continuous run results.
        eta: Learning rate.
        final_time: Final simulation time.
        i: Index or power for eta.
        folder_path: Directory to save plots.
        power: String indicating the power base for eta.
        C_theta_1ord, C_theta_2ord, C_m_1ord, C_m_2ord, C_v_1ord, C_v_2ord: Constants for plotting reference lines.
        regime: String indicating the regime type.
    Returns:
        None
    """
    title = regime + ' regime: '
    if Run_d.shape[1] == 2:
        split = Run_d.shape[1] // 2
        zeros = torch.zeros_like(Run_d[:, :split])
        plot_with_confidence_interval(Run_d[:, :split], [ Run_c_1[:, :split], Run_c_2[:, :split] ], eta, final_time, i, title+'Error for $\\theta$', folder_path = folder_path, power=power, Label = ['1-order', '2-order'], C_1ord=C_theta_1ord, C_2ord=C_theta_2ord)
        plot_with_confidence_interval(Run_d[:, split:], [ Run_c_1[:, split:], Run_c_2[:, split:] ], eta, final_time, i, title+'Error for $v$', folder_path = folder_path,  power=power, Label = ['1-order', '2-order'], C_1ord=C_v_1ord, C_2ord=C_v_2ord)
        plot_with_confidence_interval(zeros, [ Run_c_1[:, :split], Run_c_2[:, :split], Run_d[:, :split] ], eta, final_time, i, 'Dynamics for $\\theta$', folder_path = folder_path,  power=power, Label = ['1-order', '2-order', 'discrete'])
        plot_with_confidence_interval(zeros, [ Run_c_1[:, split:], Run_c_2[:, split:], Run_d[:, split:] ], eta, final_time, i, 'Dynamics for $v$', folder_path = folder_path,  power=power, Label =  ['1-order', '2-order', 'discrete'])
    elif Run_d.shape[1] == 3:
        split = Run_d.shape[1] // 3
        zeros = torch.zeros_like(Run_d[:, :split])
        plot_with_confidence_interval(Run_d[:, :split], [ Run_c_1[:, :split], Run_c_2[:, :split] ], eta, final_time, i, title+'Error for $\\theta$', folder_path = folder_path, power=power, Label = ['1-order', '2-order'], C_1ord=C_theta_1ord, C_2ord=C_theta_2ord)
        plot_with_confidence_interval(Run_d[:, split: 2*split], [ Run_c_1[:, split:2*split], Run_c_2[:, split:2*split] ], eta, final_time, i, title+'Error for $m$', folder_path = folder_path,  power=power, Label = ['1-order', '2-order'], C_1ord=C_m_1ord, C_2ord=C_m_2ord)
        plot_with_confidence_interval(Run_d[:, 2*split:], [ Run_c_1[:, 2*split:], Run_c_2[:, 2*split:] ], eta, final_time, i, title+'Error for $v$', folder_path = folder_path,  power=power, Label = ['1-order', '2-order'], C_1ord=C_v_1ord, C_2ord=C_v_2ord)
        plot_with_confidence_interval(zeros, [ Run_c_1[:, :split], Run_c_2[:, :split], Run_d[:, :split] ], eta, final_time, i, title+'Dynamics for $\\theta$', folder_path = folder_path,  power=power, Label = ['1-order', '2-order', 'discrete'])
        plot_with_confidence_interval(zeros, [ Run_c_1[:, split:2*split], Run_c_2[:, split:2*split], Run_d[:, split:2*split] ], eta, final_time, i, title+'Dynamics for $m$', folder_path = folder_path,  power=power, Label =  ['1-order', '2-order', 'discrete'])
        plot_with_confidence_interval(zeros, [ Run_c_1[:, 2*split:], Run_c_2[:, 2*split:], Run_d[:, 2*split:] ], eta, final_time, i, title+'Dynamics for $v$', folder_path = folder_path,  power=power, Label =  ['1-order', '2-order', 'discrete'])
    else:
        raise ValueError("Run_d have wrong shape")
    

def plot_with_confidence_interval(Run_d, list_run, eta, final_time, i, title, folder_path = None, power = '10', Label = None, C_1ord=1, C_2ord=1):
    """
    Plots the error with confidence intervals for different runs and saves the figure.
    Args:
        Run_d: Discrete run results.
        list_run: List of continuous run results.
        eta: Learning rate.
        final_time: Final simulation time.
        i: Index or power for eta.
        title: Plot title.
        folder_path: Directory to save plots.
        power: String indicating the power base for eta.
        Label: List of labels for the runs.
        C_1ord, C_2ord: Constants for plotting reference lines.
    Returns:
        None
    """
    def aux(run_c, run_d, x, color, label):
        Error = run_c - run_d
        plt.plot(x, make_numpy(Error), label = label, color=color)
    sns.set_style("whitegrid")
    x = np.linspace(0, final_time, Run_d.shape[0])
    plt.figure(figsize=(10, 6))
    color = ['tab:blue', 'tab:green', 'tab:orange']
    for j, r in enumerate(list_run):
        aux(r, Run_d, x, color[j], Label[j])
    if len(list_run)==2 or len(list_run)==4:
        plt.plot(x, -C_1ord * eta * np.ones_like(x), color='tab:orange', linestyle='--', label='$\\tau$')
        plt.plot(x, C_1ord * eta * np.ones_like(x), color='tab:orange', linestyle='--')
        plt.plot(x, -C_2ord * eta**2 * np.ones_like(x), color='tab:red', linestyle='--', label='$\\tau^2$')
        plt.plot(x, C_2ord * eta**2 * np.ones_like(x), color='tab:red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Error')

    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5) 

    title_cleaned = title.replace('$', '').replace('\\', '').replace(':', '_')
    if power == '10':
        path_joined = os.path.join(folder_path, f"tau_10{-i/3:.2f}_{title_cleaned}.pdf")
    elif power == '2':
        path_joined = os.path.join(folder_path, f"tau_2{-i:.2f}_{title_cleaned}.pdf")
    plt.savefig(path_joined, format='pdf', bbox_inches='tight')
    plt.close()


def plot_error_vs_eta(data, title, folder_path=None):
    """
    Plots the error versus eta for different options and saves the figure.
    Args:
        data: Dictionary of error data.
        title: Plot title.
        folder_path: Directory to save plots.
    Returns:
        None
    """
    data = {k: v for k, v in data.items() if isinstance(k, (int, float))}
    df_errors = pd.DataFrame(data, index=['Error Option 1', 'Error Option 2', 'tau', 'tau^2']).T.reset_index()

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_errors, x='index', y='Error Option 1', label='1-order', marker='o', color='tab:blue', markersize=8)
    sns.lineplot(data=df_errors, x='index', y='Error Option 2', label='2-order', marker='s', color='tab:green', markersize=8)
    sns.lineplot(data=df_errors, x='index', y='tau', label='$\\tau$', marker='', linestyle=':', color='tab:orange')
    sns.lineplot(data=df_errors, x='index', y='tau^2', label='$\\tau^2$', marker='', linestyle=':', color='tab:red')
    plt.xscale('log', base = 2)
    plt.yscale('log')
    plt.xlabel('$\\tau$', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  

    etas = list(df_errors['index'].values)
    powers = [round(torch.log2(torch.tensor(1.0 / eta)).item(), 1) for eta in etas]
    labels = [f'$2^{{{-power}}}$' for power in powers]
    plt.xticks(ticks=etas, labels=labels)
    plt.gca().invert_xaxis()

    title_cleaned = title.replace('$', '').replace('\\', '').replace(':', '_')
    path_joined = os.path.join(folder_path, title_cleaned+"_different_tau.pdf")
    plt.savefig(path_joined, format='pdf', bbox_inches='tight')
    plt.close()

def make_numpy(tensor):
    """
    Converts a PyTorch tensor to a NumPy array and squeezes it.
    Args:
        tensor: Input PyTorch tensor.
    Returns:
        NumPy array.
    """
    return tensor.detach().cpu().numpy().squeeze()


def compute_constant_for_plot(data, eta, Run_d_mean, Run_c_1_mean, Run_c_2_mean, opt = 'max'):
    """
    Computes constants for plotting error curves and updates the data dictionary.
    Args:
        data: Dictionary to update.
        eta: Learning rate.
        Run_d_mean: Mean of discrete run.
        Run_c_1_mean: Mean of first continuous run.
        Run_c_2_mean: Mean of second continuous run.
        opt: Option for 'max' or 'final' error.
    Returns:
        Updated data dictionary.
    """
    if opt == 'final':
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()]
    elif opt == 'max':
        data[eta] = [torch.max(torch.abs(Run_d_mean - Run_c_1_mean)).item(), torch.max(torch.abs(Run_d_mean - Run_c_2_mean)).item()]

    log_differences_C1 = []
    log_differences_C2 = []
    for eta, values in data.items():
        if isinstance(eta, (int, float)):
            log_diff_C1 = torch.log(torch.tensor(values[0])) - torch.log(torch.tensor(eta))
            log_differences_C1.append(log_diff_C1.item())
            log_diff_C2 = torch.log(torch.tensor(values[1])) - 2 * torch.log(torch.tensor(eta))
            log_differences_C2.append(log_diff_C2.item())
    log_C1_theta = torch.mean(torch.tensor(log_differences_C1))
    log_C2_theta = torch.mean(torch.tensor(log_differences_C2))
    C1_theta = torch.exp(log_C1_theta)
    C2_theta = torch.exp(log_C2_theta)

    if opt == 'max':                    # set 'max' to use max costants for the plot of the single variable vs time
        data['C1'] = C1_theta.item()
        data['C2'] = C2_theta.item()
    if opt == 'final':
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item(), C1_theta*eta, C2_theta*eta**2]
    elif opt == 'max':
        data[eta] = [torch.max(torch.abs(Run_d_mean - Run_c_1_mean)).item(), torch.max(torch.abs(Run_d_mean - Run_c_2_mean)).item(), C1_theta*eta, C2_theta*eta**2]    
    for prev_eta in list(data.keys()):
        if isinstance(prev_eta, (int, float)):
            data[prev_eta][2] = C1_theta.item() * prev_eta
            data[prev_eta][3] = C2_theta.item() * prev_eta**2
    return data

def Plot_error_vs_eta_aux(data_final_list, data_max_list, eta, Run_d_mean, Run_c_1_mean, Run_c_2_mean, path_folder, regime = 'Balistic'):
    """
    Helper function to plot error vs eta for all variables and save the results.
    Args:
        data_final_list: List of dictionaries for final errors.
        data_max_list: List of dictionaries for max errors.
        eta: Learning rate.
        Run_d_mean: Mean of discrete run.
        Run_c_1_mean: Mean of first continuous run.
        Run_c_2_mean: Mean of second continuous run.
        path_folder: Directory to save plots.
        regime: String indicating the regime type.
    Returns:
        Tuple of updated data_final_list and data_max_list.
    """
    if len(data_final_list) == 2:
        dict_title = {0 : '$\\theta$', 1 : '$v$'}
        dict_path_title = {0 : 'theta', 1 : 'v'}
    else:
        dict_title = {0 : '$\\theta$', 1 : '$m$', 2 : '$v$'}
        dict_path_title = {0 : 'theta', 1 : 'm', 2 : 'v'}

    text = regime + ' regime: '    
    for i, data in enumerate(data_final_list):
        data_final_list[i] = compute_constant_for_plot(data, eta, Run_d_mean[:, i], Run_c_1_mean[:, i], Run_c_2_mean[:, i], opt = 'final')
        plot_error_vs_eta(data_final_list[i], title=text + 'Final error for ' + dict_title[i], folder_path = path_folder)
        torch.save(data_final_list[i], os.path.join(path_folder, f"data_{dict_path_title[i]}_final.pt"))

    for i, data in enumerate(data_max_list):
        data_max_list[i] = compute_constant_for_plot(data, eta, Run_d_mean[:, i], Run_c_1_mean[:, i], Run_c_2_mean[:, i], opt = 'max')
        plot_error_vs_eta(data_max_list[i], title=text + 'Max error for ' + dict_title[i], folder_path = path_folder)
        torch.save(data_max_list[i], os.path.join(path_folder, f"data_{dict_path_title[i]}_max.pt"))
        
    return data_final_list, data_max_list
