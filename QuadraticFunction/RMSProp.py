import torchsde
import time
import sys
import torch
import QuadraticFunction.Function as Fun
import Algorithms.Balistic_regime_RMSProp as BR_RMSPROP
import Algorithms.Batch_eq_regime_RMSProp as BER_RMSPROP
import Algorithms.Utils as Algo_utils
import os
import time
import QuadraticFunction.utils as utils

sys.setrecursionlimit(10000)
SIGMA_VALUE = 0.25

def Dynamic_simulation(fun, eta, c, beta, skip_initial_point, step, y0, final_time, option, regularizer, Verbose = False, epsilon = 1e-3, dataset = None):
    """
    Simulates the dynamics of the RMSProp optimizer under various regimes.
    Depending on the 'option' parameter, it selects the appropriate simulation method (discrete, SDE, or ODE) and regime (balistic or batch equivalent).
    Args:
        fun: The function to optimize.
        eta: Learning rate.
        c: Constant for the optimizer.
        beta: Beta parameter for RMSProp.
        skip_initial_point: Number of initial points to skip.
        step: Number of steps in the simulation.
        y0: Initial state tensor.
        final_time: Final simulation time.
        option: String specifying the simulation regime.
        regularizer: Regularizer object.
        Verbose: If True, prints additional information.
        epsilon: Small value to avoid division by zero.
        dataset: Optional dataset for stochastic simulations.
    Returns:
        Simulation results as a tensor.
    """
    ts = torch.arange(eta* skip_initial_point , final_time, eta).to(y0.device)
    fun.change_batch_size(y0.shape[0])

    
    regularizer.set_costant( eta *torch.ones_like(y0[:, 2:]) ) 
    if option == 'RMSProp_Dyscrete_balistic_regime':
        return BR_RMSPROP.Discrete_RMProp_balistic_regime(fun, dataset, eta, beta, c, step, y0, skip = skip_initial_point, epsilon=epsilon, loss_bool=False)[:, skip_initial_point:, :]
    elif option == 'RMSProp_Dyscrete_batch_equivalent_regime':
        return BER_RMSPROP.Discrete_RMProp_batch_eq_regime(fun, dataset, eta, beta, c, step, y0, skip = skip_initial_point, epsilon=epsilon, loss_bool=False)[:, skip_initial_point:, :]
    
    elif option == 'RMSProp_SDE_1order_balistic_regime':
        sde = BR_RMSPROP.RMSprop_SDE_1order_balistic_regime(eta, c, fun, ts, regularizer, epsilon=epsilon, sigma_value=SIGMA_VALUE, Verbose=Verbose)
        return torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif option == 'RMSProp_Approx_1order_balistic_regime':
        sde = BR_RMSPROP.RMSprop_deterministic(eta, c, fun, ts, regularizer, epsilon=epsilon, sigma_value=SIGMA_VALUE, Verbose=Verbose)
        return torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif option == 'RMSProp_SDE_2order_balistic_regime':
        sde = BR_RMSPROP.RMSprop_SDE_2order_balistic_regime(eta, c, fun, ts, regularizer, epsilon=epsilon, sigma_value=SIGMA_VALUE, Verbose=Verbose)
        return torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif option == 'RMSProp_SDE_1order_batch_equivalent_regime':
        sde = BER_RMSPROP.RMSprop_SDE_1order_batch_eq_regime(eta, c, fun, ts, regularizer, epsilon=epsilon, sigma_value=SIGMA_VALUE, Verbose=Verbose)
        return torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif option == 'RMSProp_SDE_2order_batch_equivalent_regime':
        sde = BER_RMSPROP.RMSprop_SDE_2order_batch_eq_regime(eta, c, fun, ts, regularizer, epsilon=epsilon, sigma_value=SIGMA_VALUE, Verbose=Verbose)
        return torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    else:
        raise ValueError("Invalid option for simulation. Choose from 'Dyscrete_balistic_regime', 'Dyscrete_batch_equivalent_regime', 'RMSProp_SDE_1order_balistic_regime', 'RMSProp_SDE_2order_balistic_regime', 'RMSProp_SDE_1order_batch_equivalent_regime', 'RMSProp_SDE_2order_batch_equivalent_regime'.")
    

def N_simulations(fun, regularizer, y0, eta, beta, c, final_time, run, batch_size, opt_discr, opt_cont_1, opt_cont_2, opt_cont_3=None, epsilon = 1e-6, dataset =None):
    """
    Runs multiple simulations for different regimes and collects statistics (mean and variance) for each run.
    Args:
        fun: The function to optimize.
        regularizer: Regularizer object.
        y0: Initial state tensor.
        eta: Learning rate.
        beta: Beta parameter for RMSProp.
        c: Constant for the optimizer.
        final_time: Final simulation time.
        run: Tuple with number of runs for SDE1 and SDE2.
        batch_size: Tuple with batch sizes for SDE1 and SDE2.
        opt_discr: Option for discrete regime.
        opt_cont_1: Option for first continuous regime.
        opt_cont_2: Option for second continuous regime.
        opt_cont_3: Option for third continuous regime (optional).
        epsilon: Small value to avoid division by zero.
        dataset: Optional dataset for stochastic simulations.
    Returns:
        Tuple of mean results for each regime.
    """
    run_SDE_1, run_SDE_2 = run
    batch_size_SDE_1, batch_size_SDE_2 = batch_size

    step = int(torch.ceil(torch.tensor(final_time / eta)))

    skip_initial_point = 2

    Run_d_meanVar_theta = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    Run_c_1_meanVar_theta = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    Run_c_2_meanVar_theta = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    Run_d_meanVar_v = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    Run_c_1_meanVar_v = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    Run_c_2_meanVar_v = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
    if opt_cont_3 != None:
        Run_c_3_meanVar_theta = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))
        Run_c_3_meanVar_v = utils.OnlineMeanVar(step-skip_initial_point, device=torch.device("cpu"))

    print(f'Starting simulation with eta: {eta:.3f}, beta: {beta:.3f}, c: {c:.3f}')
    utils.set_seed_for_dynamic(opt_discr)
    for i in range(run_SDE_2):
        start = time.time()
        y_discr = Dynamic_simulation(fun=fun, eta=eta, c=c, beta=beta, skip_initial_point=skip_initial_point, step=step, y0=y0.expand(batch_size_SDE_2, y0.shape[0]) , final_time=final_time, option=opt_discr, regularizer=regularizer, Verbose = False, epsilon = epsilon, dataset=dataset)
        print('Run:', i+1, "/", run_SDE_2, " - Run: 0/", run_SDE_1, " discete dynamic",  f"{(time.time()-start):.2f}", " seconds", end='\r')
        initial_point = y_discr[:, 0, :] 
        y_discr = y_discr.cpu()
        Run_d_meanVar_theta.update(g_function(y_discr)[:, :, 0])
        Run_d_meanVar_v.update(g_function(y_discr)[:, :, 1])

    utils.set_seed_for_dynamic(opt_cont_2)
    for i in range(run_SDE_2):
        start = time.time()
        y_cont_2 = Dynamic_simulation(fun=fun, eta=eta, c=c, beta=beta, skip_initial_point=skip_initial_point, step=step, y0=initial_point.clone(), final_time=final_time, option=opt_cont_2, regularizer=regularizer, Verbose = False, epsilon = epsilon, dataset=dataset).permute(1, 0, 2).cpu()
        print('Run:', i+1, "/", run_SDE_2, " - Run: 0/", run_SDE_1, ' end of continuous dynamic of order 2 in ', f"{(time.time()-start):.2f}", " seconds", end='\r')        
        Run_c_2_meanVar_theta.update(g_function(y_cont_2)[:, :, 0])
        Run_c_2_meanVar_v.update(g_function(y_cont_2)[:, :, 1])
        torch.cuda.empty_cache()

    utils.set_seed_for_dynamic(opt_cont_1)
    for i in range(run_SDE_1):
        start = time.time()
        y_cont_1 = Dynamic_simulation(fun=fun, eta=eta, c=c, beta=beta, skip_initial_point=skip_initial_point, step=step, y0=initial_point[0].unsqueeze(0).expand(batch_size_SDE_1, initial_point.shape[1]), final_time=final_time, option=opt_cont_1, regularizer=regularizer, Verbose = False, epsilon = epsilon, dataset=dataset).permute(1, 0, 2).cpu()
        print('Run:', run_SDE_2, "/", run_SDE_2, " - Run: ", i+1, "/", run_SDE_1, ' end of continuous dynamic of order 1', f"{(time.time()-start):.2f}", " seconds", end='\r')
        Run_c_1_meanVar_theta.update(g_function(y_cont_1)[:, :, 0])
        Run_c_1_meanVar_v.update(g_function(y_cont_1)[:, :, 1]) 

    if opt_cont_3 != None: 
        utils.set_seed_for_dynamic(opt_cont_3)
        for i in range(run_SDE_1):
            start = time.time()
            y_cont_3 = Dynamic_simulation(fun=fun, eta=eta, c=c, beta=beta, skip_initial_point=skip_initial_point, step=step, y0=initial_point[0].unsqueeze(0).expand(batch_size_SDE_1, initial_point.shape[1]), final_time=final_time, option=opt_cont_3, regularizer=regularizer, Verbose = False, epsilon = epsilon, dataset=dataset).permute(1, 0, 2).cpu()
            print('Run:', run_SDE_2, "/", run_SDE_2, " - Run: ", i+1, "/", run_SDE_1, ' end of continuous dynamic of order 3', f"{(time.time()-start):.2f}", " seconds", end='\r')
            Run_c_3_meanVar_theta.update(g_function(y_cont_3)[:, :, 0])
            Run_c_3_meanVar_v.update(g_function(y_cont_3)[:, :, 1])

    del y_discr, y_cont_1, y_cont_2
    if opt_cont_3 != None: del y_cont_3
    torch.cuda.empty_cache()

    Run_d_mean = torch.stack((Run_d_meanVar_theta.get_mean(), Run_d_meanVar_v.get_mean()), dim=1)
    Run_c_1_mean = torch.stack((Run_c_1_meanVar_theta.get_mean(), Run_c_1_meanVar_v.get_mean()), dim=1)
    Run_c_2_mean = torch.stack((Run_c_2_meanVar_theta.get_mean(), Run_c_2_meanVar_v.get_mean()), dim=1)

    if opt_cont_3 != None:
        Run_c_3_mean = torch.stack((Run_c_3_meanVar_theta.get_mean(), Run_c_3_meanVar_v.get_mean()), dim=1)
        return Run_d_mean, Run_c_1_mean, Run_c_2_mean, Run_c_3_mean
    return Run_d_mean, Run_c_1_mean, Run_c_2_mean
def g_function(x):
    """
    Splits the input tensor into theta and v components and computes their norms.
    Args:
        x: Input tensor of shape (batch, time, features).
    Returns:
        Concatenated tensor of norms for theta and v.
    """
    split = x.shape[-1]//2
    theta = x[:, :, :split]
    v = x[:, :, split:]
    def aux(x, split):
        return torch.norm(x, dim=2)/split
    return torch.concat((aux(theta, split).unsqueeze(2), aux(v, split).unsqueeze(2)), dim = 2)


def Simulation_sde(opt_discr, opt_cont_1, opt_cont_2, opt_cont_3 = None, A=None, y0=None, beta_fun=None, path_folder = None, path_folder_for_opt_3=None, dataset=None):
    """
    Orchestrates the simulation of RMSProp optimizer for different regimes and learning rates.
    Handles data collection, result saving, and plotting.
    Args:
        opt_discr: Option for discrete regime.
        opt_cont_1: Option for first continuous regime.
        opt_cont_2: Option for second continuous regime.
        opt_cont_3: Option for third continuous regime (optional).
        A: Optional matrix for the quadratic function.
        y0: Initial state tensor.
        beta_fun: Function to compute beta given eta.
        path_folder: Path to save results.
        path_folder_for_opt_3: Path to save results for opt_3.
        dataset: Optional dataset for stochastic simulations.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2
    torch.manual_seed(0)
    if device == 'cuda':
        torch.cuda.manual_seed_all(0)

    if A is None:
        fun = Fun.Quadratic_function(dim=dim)
    else:
        fun = Fun.Quadratic_function(dim=dim, A=A)

    final_time = 10
    max_batch_size = 1200000
    max_sim = 10**7-1
    data_theta_final, data_theta_max, data_v_final, data_v_max, Res = {}, {}, {}, {}, {}
    if opt_cont_3 != None:
        data_theta_final_opt_3, data_theta_max_opt_3, data_v_final_opt_3, data_v_max_opt_3, Res_opt_3 = {}, {}, {}, {}, {}
    Res['Matrix'] = fun.A
    Res['Initial point'] = y0
    Res['Final time'] = final_time
    print('Matrix', fun.A, ' condition number:', torch.linalg.cond(fun.A).item())
    print('Initial point', y0)
    if 'balistic' in opt_discr:
        regime = 'Balistic'
    elif 'batch_equivalent' in opt_discr:
        regime = 'Batch equivalent'
    else:
        regime = 'Unknown'
    Res['Regime'] = regime

    regularizer = Algo_utils.Regularizer_ReLu()
    start_tot = time.time()
    for i in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]:
        start = time.time()
        power = '2'
        eta = 2**(-i)
        beta = beta_fun.get_beta(eta)
        c = beta_fun.get_c(eta)
        print(f"\nSimulation with eta = {eta:.3f}, beta = {beta:.3f}, c = {c:.3f}\n")
        
        run_SDE_2, batch_size_SDE_2 = utils.calculate_batch_size_and_run(eta, max_sim, max_batch_size, 2, final_time)
        run_SDE_1, batch_size_SDE_1 = utils.calculate_batch_size_and_run(eta, max_sim, max_batch_size, 1, final_time)
      
        if opt_cont_3 != None:
            Run_d_mean, Run_c_1_mean, Run_c_2_mean, Run_c_3_mean = N_simulations(fun, regularizer, y0, eta, beta, c, final_time, (run_SDE_1, run_SDE_2), (batch_size_SDE_1, batch_size_SDE_2), opt_discr, opt_cont_1, opt_cont_2, opt_cont_3, dataset=dataset)
        else:
            Run_d_mean, Run_c_1_mean, Run_c_2_mean = N_simulations(fun, regularizer, y0, eta, beta, c, final_time, (run_SDE_1, run_SDE_2), (batch_size_SDE_1, batch_size_SDE_2), opt_discr, opt_cont_1, opt_cont_2, opt_cont_3, dataset=dataset)


        data_final_list = [data_theta_final, data_v_final]
        data_max_list = [data_theta_max, data_v_max]
        data_final_list, data_max_list, Res = utils.Call_plot(data_final_list, data_max_list, Res, Run_d_mean, Run_c_1_mean, Run_c_2_mean, eta, final_time, i, start, power, path_folder = path_folder, regime=regime)

        data_theta_final, data_v_final = data_final_list
        data_theta_max, data_v_max = data_max_list

        if opt_cont_3 != None:
            data_final_list_opt_3 = [data_theta_final_opt_3, data_v_final_opt_3]
            data_max_list_opt_3 = [data_theta_max_opt_3, data_v_max_opt_3]
            data_final_list_opt_3, data_max_list_opt_3, Res_opt_3 = utils.Call_plot(data_final_list_opt_3, data_max_list_opt_3,Res_opt_3, Run_d_mean, Run_c_3_mean, Run_c_2_mean, eta, final_time, i, start, power, path_folder = path_folder_for_opt_3, regime=regime)
            data_theta_final_opt_3, data_v_final_opt_3 = data_final_list_opt_3
            data_theta_max_opt_3, data_v_max_opt_3 = data_max_list_opt_3
        Res['time elapsed for ' + str(eta)] = time.time() - start

    final_time = time.time() - start_tot
    Res['time elapsed'] = final_time
    print('Final time:', final_time)
    if path_folder is not None:
        torch.save(Res, os.path.join(path_folder, "Result.pt"))

    if opt_cont_3 != None:
        Res_opt_3['Final time'] = final_time
        Res_opt_3['Matrix'] = fun.A
        Res_opt_3['Initial point'] = y0
        Res_opt_3['Final time simulation'] = 40
        Res_opt_3['Regime'] = regime
        if path_folder_for_opt_3 is not None:
            torch.save(Res_opt_3, os.path.join(path_folder_for_opt_3, "Result_opt_3.pt"))

    utils.Graph_with_right_constants(Res, 10, path_folder, data_theta_max, data_v_max, regime = regime)
    if opt_cont_3 != None:
        utils.Graph_with_right_constants(Res_opt_3, 10, path_folder_for_opt_3, data_theta_max_opt_3, data_v_max_opt_3, regime = regime)

def multiple_simulations(Main_folder=None, opt_discr=None, opt_cont_1=None, opt_cont_2=None, opt_cont_3 = None, folder_opt_3 = None, dataset=None, initial_point=torch.tensor([10., 5.]), seed = 0):
    """
    Runs multiple sets of simulations for different fixed values of c and saves the results.
    Args:
        Main_folder: Main directory to save results.
        opt_discr: Option for discrete regime.
        opt_cont_1: Option for first continuous regime.
        opt_cont_2: Option for second continuous regime.
        opt_cont_3: Option for third continuous regime (optional).
        folder_opt_3: Directory for opt_3 results.
        dataset: Optional dataset for stochastic simulations.
        initial_point: Initial point for the simulation.
        seed: Random seed.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_point = initial_point.to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    A = torch.tensor([[1.0, 0.5], [0.5, 1.0]], device=device)
    y0 =initial_point.to(device)


    fixed_values_c = [0.5]
    for c in fixed_values_c:
        c_fun = utils.fixed_c(c)
        folder_path = os.path.join(Main_folder, f"SDE1_1approx_SDE2_2approx_c_{c}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if opt_cont_3 != None:
            folder_opt_3_joined = os.path.join(folder_opt_3, f"SDE1_1approx_SDE2_2approx_c_{c}")
            if not os.path.exists(folder_opt_3_joined):
                os.makedirs(folder_opt_3_joined)
        else:
            folder_opt_3_joined = None
        Simulation_sde(opt_discr, opt_cont_1, opt_cont_2, opt_cont_3=opt_cont_3, A=A, y0=y0, beta_fun=c_fun, path_folder=folder_path, path_folder_for_opt_3 = folder_opt_3_joined, dataset=dataset)
    

def RMSProp_balistic_regime():
    """
    Runs the RMSProp optimizer simulation in the balistic regime and saves the results.
    Returns:
        None
    """
    Main_folder = './Results_QuadraticFunction'

    dataset = Fun.create_dataset(dim=2, num_samlpes=100*40*2**5)
    dataset = SIGMA_VALUE * dataset
    folder = os.path.join(Main_folder, 'RMSProp_balistic_regime_SDE1')
    if not os.path.exists(folder): os.makedirs(folder)
    opt_discr = 'RMSProp_Dyscrete_balistic_regime'
    opt_cont_1 = 'RMSProp_SDE_1order_balistic_regime'
    opt_cont_2 = 'RMSProp_SDE_2order_balistic_regime'
    opt_cont_3 = 'RMSProp_Approx_1order_balistic_regime'
    folder_opt_3 = os.path.join(Main_folder, 'RMSProp_balistic_regime_approx1')
    if not os.path.exists(folder_opt_3): os.makedirs(folder_opt_3)
    print('\n \n Balistic regime SDE1 \n')
    multiple_simulations(folder, opt_discr, opt_cont_1, opt_cont_2, opt_cont_3=opt_cont_3, dataset = dataset, folder_opt_3=folder_opt_3)

def RMSProp_batch_equivalent_regime():
    """
    Runs the RMSProp optimizer simulation in the batch equivalent regime and saves the results.
    Returns:
        None
    """
    Main_folder = './Results_QuadraticFunction'

    dataset = Fun.create_dataset(dim=2, num_samlpes=100*40*2**5)
    dataset = SIGMA_VALUE * dataset
    folder = os.path.join(Main_folder, 'RMSProp_batch_equivalent_regime_SDE1')
    if not os.path.exists(folder): os.makedirs(folder)
    opt_discr = 'RMSProp_Dyscrete_batch_equivalent_regime'
    opt_cont_1 = 'RMSProp_SDE_1order_batch_equivalent_regime'
    opt_cont_2 = 'RMSProp_SDE_2order_batch_equivalent_regime'
    print('\n \n Batch equivalent regime SDE1 \n')
    multiple_simulations(folder, opt_discr, opt_cont_1, opt_cont_2, dataset = dataset)


if __name__ == "__main__":
    RMSProp_batch_equivalent_regime()
    RMSProp_balistic_regime()
    
