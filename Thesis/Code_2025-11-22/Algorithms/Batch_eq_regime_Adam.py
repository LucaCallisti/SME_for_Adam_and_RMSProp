import torch
import torchsde
import time
from Algorithms.Utils import Regularizer_ReLu, SDE_basic


torch.set_printoptions(precision=6)
    

class Adam_SDE_2order_batch_equivalent_regime(SDE_basic):
    def __init__(self, eta, c, function, All_time, regularizer, epsilon = 1e-6, sigma_value=0.5, constant_noise = True, Verbose = True):
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
        self.temp = 0
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
        # second_term = - self.c_2 * self.Gamma(t) * 0.5 * (denom**2) * ( 1/torch.sqrt(v_reg) ) * (self.diag_Sigma - self.v) * (self.m  * v_reg_grad)
        second_term = - self.c_2 * self.Gamma(t) * 0.5 * (denom**3) * (self.diag_Sigma - self.v) * (self.m  * v_reg_grad)
        third_term =  self.derivative_aux_b_1(t, denom) * self.m
        return  0.5 * (first_term +  second_term + third_term)
    
    def b_0_m(self):
        return self.c_1 * (self.f_grad - self.m)
    def b_1_m(self, t, denom):
        first_term = self.c_1 * self.Gamma(t) * torch.bmm(self.f_hessian , (self.m * denom).unsqueeze(2) ).squeeze(2) 
        second_term = self.c_1**2 * (self.f_grad - self.m) 
        return 0.5 * (first_term + second_term)
    
    def b_0_v(self):
        return self.c_2 * (self.diag_Sigma - self.v)
    def b_1_v(self, t, denom):
        first_term = self.c_2 * self.f_grad_square
        second_term = 0.5 * self.c_2**2 * (self.diag_Sigma - self.v)
        if self.constant_noise is False:
           temp1 = self.m * denom
           third_term = 0.5 *self.c_1 * self.c_2 *self.Gamma(t) * torch.bmm(self.grad_Sigma_diag, temp1.unsqueeze(2)).squeeze(2)
        else:
            third_term = 0
        return  (first_term + second_term + third_term)

    def g(self, t, x):
        self.divide_input(x, t)

        v_reg = self.regularizer.regulariz_function(self.v)

        denom = 1/(torch.sqrt(v_reg) + self.eps * torch.sqrt(self.gamma_2(t)))

        M_12 = -self.c_1 * self.eta / 2 * torch.bmm(torch.diag_embed(self.Gamma(t) * denom), self.Sigma_sqrt)
        # M_22 = self.c_1 * (1+ self.eta/2) *  self.Sigma_sqrt
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
    def f(self, t, x):
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  

        # Theta coefficient
        denom = 1/(torch.sqrt(self.v) + self.eps * torch.sqrt(self.gamma_2(t)))
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
        self.diffusion =  torch.cat((M_top, M_middle, M_bottom), dim=1)
        return self.diffusion

    def gamma_1(self, t):
        return 1 - torch.exp(-t * self.c_1)
    def gamma_2(self, t):
        return 1 - torch.exp(-t * self.c_2)
    def Gamma(self, t):
        return torch.sqrt(self.gamma_2(t))/self.gamma_1(t) 

def Discrete_Adam_batch_equivalent_regime(funz, noise, lr, beta, c, num_steps, x_0, skip, epsilon = 1e-6, verbose = False, loss_bool = True):
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

    lr = torch.tensor(lr)
    temp = 0
    max_lenghth_gamma_list = 1000
    noise_shuffled = noise[torch.randperm(noise.shape[0])]

    print('Starting point:', path_x[:,0,:].mean().item())

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
        m = path_m[:, step]
        if loss_bool:
            Loss_values[:, step] = funz.loss_batch(x)
        g = funz.noisy_grad_batcheq(x, gamma, lr)

        gamma_1 = 1-beta_1**(step+2) * torch.ones_like(v)
        sqrt_gamma_2 = torch.sqrt(torch.tensor(1-beta_2**(step+1) )) * torch.ones_like(v)

        path_v[:, step+1] = beta_2 * v + lr**2 * c_2 * torch.pow(g, 2)
        path_m[:, step+1] = beta_1 * m + lr * c_1 * g

        path_x[:, step+1] = x - lr * sqrt_gamma_2 / ( (torch.sqrt(v) + epsilon * sqrt_gamma_2) * gamma_1) * (path_m[:, step+1])  
        if verbose and step*lr >= temp:
            temp += 0.1
        
    if loss_bool:
        Loss_values[:, -1] = funz.loss_batch(path_x[:, -1])
        return torch.concat((path_x, path_m, path_v), dim = 2), Loss_values
    return torch.concat((path_x, path_m, path_v), dim = 2)