import torch
import time
from Algorithms.Utils import SDE_basic


torch.set_printoptions(precision=6)
    

class Adam_SDE_2order_batch_equivalent_regime(SDE_basic):
    """
    Implements the 2nd order Adam optimizer in the batch equivalent regime as an SDE.
    """
    @torch.no_grad()
    def __init__(self, eta, c, function, All_time, regularizer, epsilon = 1e-6, sigma_value=0.5, constant_noise = True, Verbose = True):
        """
        Initialize the Adam SDE for the batch equivalent regime.
        Args:
            eta: Learning rate.
            c: Tuple of coefficients.
            function: Target function.
            All_time: Array of time points.
            regularizer: Regularizer object.
            epsilon: Small value to avoid division by zero.
            sigma_value: Noise scale.
            constant_noise: If noise is constant.
            Verbose: Verbosity flag.
        """
        super().__init__(noise_type="general")
        self.regime = 'batch_equivalent'
        self.eq = 'Adam'
        self.constant_noise = constant_noise

        self.sigma_value = sigma_value
        c_1, c_2 = c

        self.eta = torch.tensor(eta)
        self.c_1 = c_1
        self.c_2 = c_2
        self.sum_1 =  (self.c_1 + self.c_1**2 * self.eta / 2 + self.c_1 ** 3 * self.eta**2 / 6 + self.c_1**4 * self.eta**3 / 24 + self.c_1**5 * self.eta**4 / 120)
        self.sum_2 =  (self.c_2 + self.c_2**2 * self.eta / 2 + self.c_2 ** 3 * self.eta**2 / 6 + self.c_2**4 * self.eta**3 / 24 + self.c_2**5 * self.eta**4 / 120)
        self.eps = epsilon 
        self.fun = function

        self.theta_old = None
        self.diffusion = None
        self.drift = None
        self.i = 1
        self.verbose = Verbose
        self.final_time = All_time[-1]
        self.All_time = All_time
        self.t_nan, self.t_verbose = 0, 0
        self.regularizer = regularizer
    
    @torch.no_grad()
    def f(self, t, x):
        """
        Compute the drift term of the SDE at time t and state x.
        """
        self.divide_input(x, t)
        
        v_reg = self.regularizer.regulariz_function(self.v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(self.v)

        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))
        coef_theta = self.b_0_theta(t, denom) + self.eta * self.b_1_theta(t, denom, v_reg, v_reg_grad)

        coef_m = self.b_0_m() + self.eta * self.b_1_m(t, denom)
    
        coef_v = self.b_0_v() + self.eta * self.b_1_v(t, denom)

        self.drift = torch.concat((coef_theta, coef_m, coef_v), dim = 1)
        self.Verbose(t)       
        self.is_it_Nan(self.drift, x, t, 'drift 2 order')

        return self.drift
    
    @torch.no_grad()
    def gamma_1(self, t):
        """
        Compute the gamma_1 coefficient for the SDE.
        """
        return 1 - torch.exp(-(t+self.eta) * self.sum_1) 
    @torch.no_grad()
    def gamma_2(self, t):
        """
        Compute the gamma_2 coefficient for the SDE.
        """
        return 1 - torch.exp(-t * self.sum_2)
    @torch.no_grad()
    def derivative_gamma_1(self, t):
        """
        Compute the derivative of gamma_1 with respect to t.
        """
        return self.sum_1 * torch.exp(-(t+self.eta) * self.sum_1)
    @torch.no_grad()
    def derivative_gamma_2(self, t):
        """
        Compute the derivative of gamma_2 with respect to t.
        """
        return self.sum_2 * torch.exp(-t * self.sum_2)
    @torch.no_grad()
    def Gamma(self, t):
        """
        Compute the Gamma coefficient for the SDE.
        """
        return torch.sqrt(self.gamma_2(t))/self.gamma_1(t) 
    
    @torch.no_grad()
    def derivative_Gamma(self, t):
        """
        Compute the derivative of Gamma with respect to t.
        """
        return (0.5 * self.derivative_gamma_2(t) * self.gamma_1(t) / torch.sqrt(self.gamma_2(t)) - torch.sqrt(self.gamma_2(t)) * self.derivative_gamma_1(t)) / (self.gamma_1(t)**2)
    @torch.no_grad()
    def derivative_aux_b_1(self, t, denom):
        """
        Auxiliary derivative for b_1_theta calculation.
        """
        return self.Gamma(t) * (0.5 * self.derivative_gamma_2(t) * denom**2 /  torch.sqrt(self.gamma_2(t)) ) + self.derivative_Gamma(t) * denom

    @torch.no_grad()
    def b_0_theta(self, t, denom):
        """
        Compute the 0th order theta coefficient.
        """
        return - self.Gamma(t) * self.m  * denom
    @torch.no_grad()
    def b_1_theta(self, t, denom, v_reg, v_reg_grad):
        """
        Compute the 1st order theta coefficient.
        """
        first_term = self.c_1 * self.Gamma(t) * (self.m - self.f_grad) * denom
        second_term = - self.c_2 * self.Gamma(t) * 0.5 * (denom**3) * (self.diag_Sigma - self.v) * (self.m  * v_reg_grad)
        third_term =  self.derivative_aux_b_1(t, denom) * self.m
        return  0.5 * (first_term +  second_term + third_term)
    
    @torch.no_grad()
    def b_0_m(self):
        """
        Compute the 0th order m coefficient.
        """
        return self.c_1 * (self.f_grad - self.m)
    @torch.no_grad()
    def b_1_m(self, t, denom):
        """
        Compute the 1st order m coefficient.
        """
        first_term = self.c_1 * self.Gamma(t) * torch.bmm(self.f_hessian , (self.m * denom).unsqueeze(2) ).squeeze(2) 
        second_term = self.c_1**2 * (self.f_grad - self.m) 
        return 0.5 * (first_term + second_term)
    
    @torch.no_grad()
    def b_0_v(self):
        """
        Compute the 0th order v coefficient.
        """
        return self.c_2 * (self.diag_Sigma - self.v)
    @torch.no_grad()
    def b_1_v(self, t, denom):
        """
        Compute the 1st order v coefficient.
        """
        first_term = self.c_2 * self.f_grad_square
        second_term = 0.5 * self.c_2**2 * (self.diag_Sigma - self.v)
        if self.constant_noise is False:
           temp1 = self.m * denom
           third_term = 0.5 *self.c_1 * self.c_2 *self.Gamma(t) * torch.bmm(self.grad_Sigma_diag, temp1.unsqueeze(2)).squeeze(2)
        else:
            third_term = 0
        return  (first_term + second_term + third_term)

    @torch.no_grad()
    def g(self, t, x):
        """
        Compute the diffusion term of the SDE at time t and state x.
        """
        self.divide_input(x, t)

        v_reg = self.regularizer.regulariz_function(self.v)

        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))

        M_12 = - self.eta / 2 * torch.bmm(torch.diag_embed(self.Gamma(t) * denom), self.Sigma_sqrt)
        M_22 = self.c_1 * self.Sigma_sqrt
        if self.constant_noise is False:
            temp1 = self.m * denom
            temp2 = torch.einsum('bi,bijk->bijk', temp1, self.grad_Sigma)
            temp3 = temp2.sum(dim = 1)  
            M_22 += self.c_1 * self.eta /2 * self.Gamma(t) * temp3 
        M_32 = -2* self.eta *self.c_2*torch.bmm(torch.diag_embed(self.f_grad), self.Sigma_sqrt)
        M_33 = self.c_2 * torch.sqrt(self.eta) *self.square_root_var_z_squared
        
        M_11 = torch.zeros_like(M_22)
        M_13 = torch.zeros_like(M_22)
        M_21 = torch.zeros_like(M_22)
        M_23 = torch.zeros_like(M_22)
        M_31 = torch.zeros_like(M_22)

        M_top = torch.cat((M_11, M_12, M_13), dim=2)
        M_middle = torch.cat((M_21, M_22, M_23), dim=2)
        M_bottom = torch.cat((M_31, M_32, M_33), dim=2)
        self.diffusion = torch.cat((M_top, M_middle, M_bottom), dim=1)
     
        return self.diffusion
        
    
class Adam_SDE_1order_batch_equivalent_regime(Adam_SDE_2order_batch_equivalent_regime):
    """
    Implements the 1st order Adam optimizer in the batch equivalent regime as an SDE.
    """
    @torch.no_grad()
    def f(self, t, x):
        """
        Compute the drift term for the 1st order SDE at time t and state x.
        """
        self.divide_input(x, t)
  

        denom = 1/(torch.sqrt(self.v) + self.eps * torch.sqrt(self.gamma_2(t)))
        coef_theta = self.b_0_theta(t, denom)
        coef_m = self.b_0_m()
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_m, coef_v), dim = 1)

        self.Verbose(t)  
        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        return self.drift

    @torch.no_grad()
    def g(self, t, x):
        """
        Compute the diffusion term for the 1st order SDE at time t and state x.
        """
        self.divide_input(x, t)

        M_22 = self.c_1 * self.Sigma_sqrt

        M_32 = torch.zeros_like(M_22)
        M_33 = torch.zeros_like(M_22)
        M_11 = torch.zeros_like(M_22)
        M_12 = torch.zeros_like(M_22)
        M_13 = torch.zeros_like(M_22)
        M_21 = torch.zeros_like(M_22)
        M_23 = torch.zeros_like(M_22)
        M_31 = torch.zeros_like(M_22)

        M_top = torch.cat((M_11, M_12, M_13), dim=2)
        M_middle = torch.cat((M_21, M_22, M_23), dim=2)
        M_bottom = torch.cat((M_31, M_32, M_33), dim=2)
        self.diffusion =  torch.cat((M_top, M_middle, M_bottom), dim=1)
        return self.diffusion

    @torch.no_grad()
    def gamma_1(self, t):
        """
        Compute the gamma_1 coefficient for the SDE (1st order).
        """
        return 1 - torch.exp(-t * self.c_1)
    @torch.no_grad()
    def gamma_2(self, t):
        """
        Compute the gamma_2 coefficient for the SDE (1st order).
        """
        return 1 - torch.exp(-t * self.c_2)
    @torch.no_grad()
    def Gamma(self, t):
        """
        Compute the Gamma coefficient for the SDE (1st order).
        """
        return torch.sqrt(self.gamma_2(t))/self.gamma_1(t) 

@torch.no_grad()
def Discrete_Adam_batch_equivalent_regime(funz, noise, tau, beta, c, num_steps, x_0, skip, epsilon = 1e-6, verbose = False, loss_bool = True):
    """
    Discrete implementation of the Adam optimizer in the batch equivalent regime.
    Args:
        funz: Function object with required methods.
        noise: Noise tensor.
        tau: Step size.
        beta: Tuple of beta coefficients.
        c: Tuple of coefficients.
        num_steps: Number of steps.
        x_0: Initial state.
        skip: Number of initial steps to skip noise.
        epsilon: Small value to avoid division by zero.
        verbose: Verbosity flag.
        loss_bool: Whether to compute and return loss values.
    Returns:
        Tuple of (paths, loss values) or just paths.
    """
    beta_1, beta_2 = beta
    c_1, c_2 = c

    batch_size = x_0.shape[0]
    path_x = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    path_v = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    path_m = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    if loss_bool:
        Loss_values = torch.zeros(batch_size, num_steps, device=x_0.device)
    path_x[:, 0, :] = x_0.detach().clone()
    path_v[:, 0, :] = torch.ones_like(x_0)
    path_m[:, 0, :] = torch.zeros_like(x_0)

    tau = torch.tensor(tau)
    temp = 0
    max_lenghth_gamma_list = 1000
    noise_shuffled = noise[torch.randperm(noise.shape[0])]

    print('Starting point:', path_x[:,0,:].mean().item())
    start = time.time()
    for step in range(num_steps-1):

        if step % max_lenghth_gamma_list == 0:            
            indices = torch.randint(0, noise_shuffled.shape[0], (batch_size, max_lenghth_gamma_list))
            gamma_list = noise_shuffled[indices].to(x_0.device)
        
        if step < skip:
            gamma = torch.zeros((batch_size, x_0.shape[1]), device=x_0.device)
        else:
            step_idx = step % max_lenghth_gamma_list
            gamma = gamma_list[:, step_idx, :]

        x = path_x[:, step]
        v = path_v[:, step]
        m = path_m[:, step]
        if loss_bool:
            Loss_values[:, step] = funz.loss_batch(x)

        grad = funz.noisy_grad_batcheq(x, gamma, tau)

        gamma_1 = 1-beta_1**(step+2) * torch.ones_like(v)
        sqrt_gamma_2 = torch.sqrt(torch.tensor(1-beta_2**(step+1) )) * torch.ones_like(v)

        path_v[:, step+1] = beta_2 * v + tau**2 * c_2 * torch.pow(grad, 2)
        path_m[:, step+1] = beta_1 * m + tau * c_1 * grad
        path_x[:, step+1] = x - tau * sqrt_gamma_2 / ( (torch.sqrt(v) + epsilon * sqrt_gamma_2) * gamma_1) * (path_m[:, step+1])  
        
        if step % 10000 == 0:
            print(f'time between 10000 steps: {time.time() - start:.2f} seconds at time {step * tau:.2f}')
            start = time.time()

        if verbose and step*tau >= temp:
            temp += 1
            print(f'Time: {step*tau:.2f}, x mean: {path_x[:,step+1,:].mean().item():.4f}, v mean: {v.mean().item():.4f}, grad mean: {grad.mean().item():.4f}, m mean: {path_m[:, step+1].mean().item():.4f}, gamma1: {gamma_1.mean().item():.4f}, sqrt_gamma2: {sqrt_gamma_2.mean().item():.4f}')

    if loss_bool:
        Loss_values[:, -1] = funz.loss_batch(path_x[:, -1])
        return torch.concat((path_x, path_m, path_v), dim = 2), Loss_values
    return torch.concat((path_x, path_m, path_v), dim = 2)