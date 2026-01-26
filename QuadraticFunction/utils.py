import torch
import os
import random
import time
import QuadraticFunction.Plot as my_plot



def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across Python, PyTorch, and CUDA.
    Args:
        seed: Integer seed value.
    Returns:
        None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def set_seed_for_dynamic(option):
    """
    Sets the random seed based on the simulation option for reproducibility.
    Args:
        option: String specifying the simulation regime.
    Returns:
        None
    """
    if option == 'RMSProp_Dyscrete_balistic_regime':
        set_seed(100)
    elif option == 'RMSProp_Dyscrete_batch_equivalent_regime':
        set_seed(1100)
    elif option == 'RMSProp_SDE_1order_balistic_regime':
        set_seed(200)
    elif option == 'RMSProp_Approx_1order_balistic_regime':
        set_seed(300)
    elif option == 'RMSProp_SDE_2order_balistic_regime':
        set_seed(400)
    elif option == 'RMSProp_SDE_1order_batch_equivalent_regime':
        set_seed(1200)
    elif option == 'RMSProp_SDE_2order_batch_equivalent_regime':
        set_seed(1300)


    elif option == 'Adam_dyscrete_balistic_regime':
        set_seed(2100)
    elif option == 'Adam_dyscrete_batch_equivalent_regime':
        set_seed(3100)
    elif option == 'Adam_SDE_1order_balistic_regime':
        set_seed(2200)
    elif option == 'Adam_Approx_1order_balistic_regime':
        set_seed(2300)
    elif option == 'Adam_SDE_2order_balistic_regime':
        set_seed(2400)
    elif option == 'Adam_SDE_1order_batch_equivalent_regime':
        set_seed(3200)
    elif option == 'Adam_SDE_2order_batch_equivalent_regime':
        set_seed(3300)
    else:
        raise ValueError("Invalid option for simulation.")

def Call_plot(data_final_list, data_max_list, Res, Run_d_mean, Run_c_1_mean, Run_c_2_mean, eta, final_time, i, start, power, path_folder = None, regime = 'Balistic'):
    """
    Calls the plotting functions to generate and save plots for simulation results.
    Args:
        data_final_list: List of dictionaries for final errors.
        data_max_list: List of dictionaries for max errors.
        Res: Results dictionary to update.
        Run_d_mean: Mean of discrete run.
        Run_c_1_mean: Mean of first continuous run.
        Run_c_2_mean: Mean of second continuous run.
        eta: Learning rate.
        final_time: Final simulation time.
        i: Index or power for eta.
        start: Start time for timing.
        power: String indicating the power base for eta.
        path_folder: Directory to save plots.
        regime: String indicating the regime type.
    Returns:
        Tuple of updated data_final_list, data_max_list, and Res.
    """
    data_final_list, data_max_list = my_plot.Plot_error_vs_eta_aux(data_final_list, data_max_list, eta, Run_d_mean, Run_c_1_mean, Run_c_2_mean, path_folder = path_folder, regime = regime)

    my_plot.plot_v_and_theta_single_eta(Run_d_mean, Run_c_1_mean, Run_c_2_mean,  eta, final_time, i, folder_path = path_folder, power=power, regime = regime)
    print(f"Time elapsed: {(time.time() - start):.2f} seconds")
    
    res = {'discrete' : Run_d_mean, 'SDE_1' : Run_c_1_mean, 'SDE_2' : Run_c_2_mean}
    Res[eta] = res
    torch.save(Res, os.path.join(path_folder, "Result.pt"))
    return data_final_list, data_max_list, Res

def Graph_with_right_constants(Res, final_time, folder_path,  data_theta, data_v, data_m=None, regime = 'Balistic'):
    """
    Plots the results with the correct constants for each variable and saves the figures.
    Args:
        Res: Results dictionary containing simulation data.
        final_time: Final simulation time.
        folder_path: Directory to save plots.
        data_theta: Dictionary of theta errors and constants.
        data_v: Dictionary of v errors and constants.
        data_m: Dictionary of m errors and constants (optional).
        regime: String indicating the regime type.
    Returns:
        None
    """
    print('Graph_with_right_constants')
    C_1ord_theta = data_theta['C1']
    C_2ord_theta = data_theta['C2']
    C_1ord_v = data_v['C1']
    C_2ord_v = data_v['C2']
    if data_m != None:
        C_1ord_m = data_m['C1']
        C_2ord_m = data_m['C2']
    else:
        C_1ord_m = None
        C_2ord_m = None
    numeric_keys = [key for key in Res.keys() if isinstance(key, (int, float))]
    for i, key in enumerate(numeric_keys):
        Run_d_mean = Res[key]['discrete']
        Run_c_1_mean = Res[key]['SDE_1']
        Run_c_2_mean = Res[key]['SDE_2']
        eta = key
        power = round(torch.log2(torch.tensor(1.0 / eta)).item(), 1)
        my_plot.plot_v_and_theta_single_eta(Run_d_mean, Run_c_1_mean, Run_c_2_mean,  eta, final_time, power, folder_path = folder_path, power='2', C_theta_1ord = C_1ord_theta, C_theta_2ord = C_2ord_theta,  C_m_1ord = C_1ord_m, C_m_2ord = C_2ord_m,  C_v_1ord = C_1ord_v, C_v_2ord = C_2ord_v, regime = regime)





def calculate_batch_size_and_run(eta, max_sim, max_batch_size, order_of_approximation, final_time):
    """
    Calculates the batch size and number of runs needed for a given simulation configuration.
    Args:
        eta: Learning rate.
        max_sim: Maximum number of simulations allowed.
        max_batch_size: Maximum batch size allowed.
        order_of_approximation: Order of the approximation (1 or 2).
        final_time: Final simulation time.
    Returns:
        Tuple of (number of runs, batch size).
    """
    if order_of_approximation == 1:
        sim = max(int(100*(final_time**0.5)*eta**(-2)), 1000)
    elif order_of_approximation == 2:
        sim = max(int((final_time**0.5)*eta**(-4)), 1000)

    if sim > max_sim:
        print('Warning: number of simulation for SDE ', order_of_approximation, ' too high', sim,  'setting sim to max_sim:', max_sim+1)
        sim = max_sim
    else:
        print('Number of simulations for SDE ', order_of_approximation, ': ', sim)
    run = int((sim / max_batch_size) +1)
    batch_size = int((sim / run) +1)
    print(f"For order {order_of_approximation} approximatin batch size: {batch_size}, run: {run}")
    return run, batch_size


class OnlineMeanVar():
    """
    Computes the online mean and variance for batches of data.
    Useful for tracking statistics during simulations.
    """
    def __init__(self, dim, device=None):
        """
        Initializes the OnlineMeanVar object.
        Args:
            dim: Dimension of the data.
            device: Torch device to use (optional).
        """
        self.d = dim
        self.device = device if device else torch.device("cpu")

        self.mean = torch.zeros(self.d, device=self.device)
        self.M2 = torch.zeros(self.d, device=self.device)
        self.count = 0

    def update(self, batch):
        """
        Updates the running mean and variance with a new batch of data.
        Args:
            batch: Batch tensor of shape (B, dim).
        Returns:
            None
        """
        if batch.dim() != 2 or batch.shape[1] != self.d:
            raise ValueError(f"Expected batch shape (B, {self.d}), got {batch.shape}")

        B = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        total_count = self.count + B

        delta = batch_mean - self.mean

        self.mean += delta * B / total_count
        self.M2 += batch_var * B + (delta ** 2) * self.count * B / total_count
        self.count = total_count

    def get_mean(self):
        """
        Returns the current mean.
        Returns:
            Mean tensor.
        """
        return self.mean
    def get_var(self, unbiased=True):
        """
        Returns the current variance.
        Args:
            unbiased: Whether to use unbiased estimation.
        Returns:
            Variance tensor.
        """
        return self.M2 / (self.count - 1) if unbiased else self.M2 / self.count
    def get_std(self, unbiased=True):
        """
        Returns the current standard deviation.
        Args:
            unbiased: Whether to use unbiased estimation.
        Returns:
            Standard deviation tensor.
        """
        var = self.get_var(unbiased=unbiased)
        return torch.sqrt(var)

class fixed_c():
    """
    Helper class to provide fixed c and beta values for simulations.
    """
    def __init__(self, c):
        """
        Initializes the fixed_c object with a constant value.
        Args:
            c: Constant value for c.
        """
        self.c = c
    def get_beta(self, eta):
        """
        Returns the beta value for a given eta.
        Args:
            eta: Learning rate.
        Returns:
            Beta value.
        """
        return 1-self.c * eta
    def get_c(self, eta):
        """
        Returns the fixed c value (independent of eta).
        Args:
            eta: Learning rate (unused).
        Returns:
            Fixed c value.
        """
        return self.c
