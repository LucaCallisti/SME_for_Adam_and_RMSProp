import torch
import torchsde
import time
from Algorithms.Utils import Regularizer_ReLu, SDE_basic

torch.set_printoptions(precision=6)
    
class Adam_SDE_2order_balistic_regime(SDE_basic):
    def __init__(self, eta, c, function, All_time, regularizer, epsilon = 1e-6, sigma_value=0.5, constant_noise = True, Verbose = True):
        super().__init__(noise_type="general")
        self.regime = 'balistic'
        self.eq = 'Adam'
        if constant_noise is False:
            print("constant_noise must be True in balistic regime. Setting it to True.")
        self.constant_noise = True # useless for balistic regime 

        c_1, c_2 = c
        self.sigma_value = sigma_value

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
    
    def f(self, t, x):
        self.divide_input(x, t)
        self.Verbose(t)       
        
        v_reg = self.regularizer.regulariz_function(self.v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(self.v)

        # Theta coefficient
        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))
        coef_theta = self.b_0_theta(t, denom) + self.eta * self.b_1_theta(t, denom, v_reg, v_reg_grad)
        coef_m = self.b_0_m() + self.eta * self.b_1_m(t, denom)    
        coef_v = self.b_0_v() + self.eta * self.b_1_v(t, denom)

        self.drift = torch.concat((coef_theta, coef_m, coef_v), dim = 1)
        self.is_it_Nan(self.drift, x, t, 'drift 2 order')

        return self.drift
    
    def gamma_1(self, t):
        return 1 - torch.exp(-(t+self.eta) * self.sum_1) 
    def gamma_2(self, t):
        return 1 - torch.exp(-t * self.sum_2)
    def derivative_gamma_1(self, t):
        return self.sum_1 * torch.exp(-(t+self.eta) * self.sum_1)
    def derivative_gamma_2(self, t):
        return self.sum_2 * torch.exp(-t * self.sum_2)
    def Gamma(self, t):
        return torch.sqrt(self.gamma_2(t))/self.gamma_1(t) 
    
    def derivative_Gamma(self, t):
        return (0.5 * self.derivative_gamma_2(t) * self.gamma_1(t) / torch.sqrt(self.gamma_2(t)) - torch.sqrt(self.gamma_2(t)) * self.derivative_gamma_1(t)) / (self.gamma_1(t)**2)
    def derivative_aux_b_1(self, t, denom):
        return self.Gamma(t) * (0.5 * self.derivative_gamma_2(t) * denom**2 /  torch.sqrt(self.gamma_2(t)) ) + self.derivative_Gamma(t) * denom

    def b_0_theta(self, t, denom):
        return - self.Gamma(t) * self.m  * denom
    def b_1_theta(self, t, denom, v_reg, v_reg_grad):
        first_term = self.c_1 * self.Gamma(t) * (self.m - self.f_grad) * denom
        # second_term = - self.c_2 * self.Gamma(t) * 0.5 * (denom**2) * ( 1/torch.sqrt(v_reg) ) * (self.f_grad_square + self.diag_Sigma - self.v) * (self.m  * v_reg_grad)
        second_term = - self.c_2 * self.Gamma(t) * 0.5 * (denom**3) * (self.f_grad_square + self.diag_Sigma - self.v) * (self.m  * v_reg_grad)
        third_term = + self.derivative_aux_b_1(t, denom) * self.m

        return  0.5 * (first_term + second_term + third_term)
    
    def b_0_m(self):
        return self.c_1 * (self.f_grad - self.m)
    def b_1_m(self, t, denom):
        first_term = self.c_1 * self.Gamma(t) * torch.bmm(self.f_hessian , (self.m * denom).unsqueeze(2) ).squeeze(2) 
        second_term = self.c_1**2 * (self.f_grad - self.m) 
        return 0.5 * (first_term + second_term)
    
    def b_0_v(self):
        return self.c_2 * (self.f_grad_square + self.diag_Sigma - self.v)
    def b_1_v(self, t, denom):
        first_term = 2*self.c_2 * torch.bmm( torch.diag_embed(self.f_grad), self.f_hessian)
        first_term = self.Gamma(t) * torch.bmm(first_term, (self.f_grad*denom).unsqueeze(2)).squeeze(2)
        second_term = self.c_2**2 * (self.f_grad_square + self.diag_Sigma - self.v)
        return 0.5 * (first_term + second_term)

    def g(self, t, x):
        self.divide_input(x, t)

        M_22 = self.c_1 * self.Sigma_sqrt
        M_32 = -2*self.c_2*torch.bmm(torch.diag_embed(self.f_grad), self.Sigma_sqrt)
        M_33 = self.c_2 *self.square_root_var_z_squared
        
        M_11 = torch.zeros_like(M_22)
        M_12 = torch.zeros_like(M_22)
        M_13 = torch.zeros_like(M_22)
        M_21 = torch.zeros_like(M_22)
        M_23 = torch.zeros_like(M_22)
        M_31 = torch.zeros_like(M_22)

        M_top = torch.cat((M_11, M_12, M_13), dim=2)
        M_middle = torch.cat((M_21, M_22, M_23), dim=2)
        M_bottom = torch.cat((M_31, M_32, M_33), dim=2)
        self.diffusion = torch.sqrt(self.eta) * torch.cat((M_top, M_middle, M_bottom), dim=1)
     
        return self.diffusion
  

class Adam_SDE_1order_balistic_regime(Adam_SDE_2order_balistic_regime):
    def f(self, t, x):
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  

        # Theta coefficient
        v_reg = self.regularizer.regulariz_function(self.v)
        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))
        coef_theta = self.b_0_theta(t, denom)
        coef_m = self.b_0_m()
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_m, coef_v), dim = 1)

        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        return self.drift

    def g(self, t, x):
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
        self.diffusion = torch.sqrt(self.eta) * torch.cat((M_top, M_middle, M_bottom), dim=1)
        return self.diffusion

class Adam_deterministic(Adam_SDE_2order_balistic_regime):
    def f(self, t, x):
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  

        # Theta coefficient
        v_reg = self.regularizer.regulariz_function(self.v)
        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))
        coef_theta = self.b_0_theta(t, denom)
        coef_m = self.b_0_m()
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_m, coef_v), dim = 1)

        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        return self.drift

    def g(self, t, x):
        self.diffusion = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)  
        return self.diffusion
    
def Discrete_Adam_balistic_regime(funz, noise, tau, beta, c, num_steps, x_0, skip, epsilon = 1e-6, verbose = False, loss_bool = True):
    
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

    temp = 0
    max_lenghth_gamma_list = 1000
    noise_shuffled = noise[torch.randperm(noise.shape[0])]
    start = time.time()
    for step in range(num_steps-1):

        if step % max_lenghth_gamma_list == 0:            
            indices = torch.randint(0, noise_shuffled.shape[0], (batch_size, max_lenghth_gamma_list))
            gamma_list = noise_shuffled[indices].to(x_0.device)
        
        if step < skip:
            gamma = torch.zeros((batch_size, x_0.shape[1]), device=x_0.device)
        else:
            # Seleziona il rumore appropriato per ogni batch
            step_idx = step % max_lenghth_gamma_list
            gamma = gamma_list[:, step_idx, :]

        x = path_x[:, step]
        v = path_v[:, step]
        if loss_bool:
            Loss_values[:, step] = funz.loss_batch(x)
        grad = funz.noisy_grad_balistic(x, gamma)


        gamma_1 = 1-beta_1**(step+2) * torch.ones_like(v)
        sqrt_gamma_2 = torch.sqrt(torch.tensor(1-beta_2**(step+1) )) * torch.ones_like(v)

        path_v[:, step+1] = beta_2 * v + (1 - beta_2) * grad**2
        path_m[:, step+1] = beta_1 * path_m[:, step] + (1 - beta_1) * grad
        path_x[:, step+1] = x - tau * sqrt_gamma_2 / ( (torch.sqrt(v) + epsilon * sqrt_gamma_2) * gamma_1) * (path_m[:, step+1])

        if step % 10000 == 0:
            print(f'time between 10000 steps: {time.time() - start:.2f} seconds at time {step * tau:.2f}')
            start = time.time()

        if verbose and step*tau >= temp:
            temp += 1
            print(f'Time: {step*tau:.2f}, Current mean position: {path_x[:,step+1,:].mean().item()}, v mean: {path_v[:,step+1,:].mean().item()}, grad mean: {grad.mean().item()}')
    
    if loss_bool:
        Loss_values[:, -1] = funz.loss_batch(path_x[:, -1])
        return torch.concat((path_x, path_m, path_v), dim = 2), Loss_values
    else:
        return torch.concat((path_x, path_m, path_v), dim = 2)