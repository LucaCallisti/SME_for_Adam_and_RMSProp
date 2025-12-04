import torch
import torchsde
import time
import sys
from Algorithms.Utils import Regularizer_ReLu, SDE_basic

sys.setrecursionlimit(2000)

torch.set_printoptions(precision=6)
    

class RMSprop_SDE_2order_balistic_regime(SDE_basic):
    def __init__(self, tau, c, function, All_time, regularizer, epsilon = 1e-6, sigma_value = 1.0, constant_noise = True, Verbose = False):
        super().__init__(noise_type="general")
        self.regime = 'balistic'
        self.eq = 'RMSProp'
        if constant_noise is False:
            print("constant_noise must be True in balistic regime. Setting it to True.")
        self.costant_noise = True # useless for balistic regime 


        self.sigma_value = sigma_value

        self.tau = torch.tensor(tau)
        self.c = c
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
        # print(f"t: {t:.2f}, x: {x}")
        
        self.divide_input(x, t)
        self.Verbose(t)       
        
        v_reg = self.regularizer.regulariz_function(self.v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(self.v)

        denom = 1/(torch.sqrt(v_reg) + self.eps)
        coef_theta = self.b_0_theta(denom) + self.tau * self.b_1_theta(denom, v_reg, v_reg_grad)

        # print(f't: {t:.2f}, theta: {self.theta[0, :3]}, v: {self.v[0, :3]}, v_reg: {v_reg[0, :3]}   , eps: {self.eps}')
        # print(f't: {t:.2f}, b_0: { self.b_0_theta(denom)[0, :3]}, grad: {self.f_grad[0, :3]}, denom: {1/denom[0, :3]}, coef_theta: {coef_theta[0, :3]}')
    
        coef_v = self.b_0_v() + self.tau * self.b_1_v(denom)

        self.drift = torch.concat((coef_theta, coef_v), dim = 1)
        self.is_it_Nan(self.drift, x, t, 'drift 2 order')

        return self.drift

    def b_0_theta(self, denom):
        return - self.f_grad  * denom
    def b_1_theta(self, denom, v_reg, v_reg_grad):
        OuterProduct = torch.einsum('ki,kj->kij', denom, denom)
        term_b_1_theta = torch.bmm(self.f_hessian*OuterProduct, self.f_grad.unsqueeze(2)).squeeze(2)+ self.c* 0.5* (denom**3)  *(self.f_grad_square + self.diag_Sigma - self.v) * (self.f_grad  * v_reg_grad)
        # term_b_1_theta = torch.bmm(self.f_hessian*OuterProduct, self.f_grad.unsqueeze(2)).squeeze(2)+ self.c* 0.5* (denom**2) * (1/torch.sqrt(v_reg)) *(self.f_grad_square + self.diag_Sigma - self.v) * (self.f_grad  * v_reg_grad)

        return  - 0.5 * term_b_1_theta
    def b_0_v(self):
        return self.c * (self.f_grad_square + self.diag_Sigma - self.v)
    def b_1_v(self, denom):
        first_term = 2*self.c * torch.bmm( torch.diag_embed(self.f_grad), self.f_hessian)
        first_term = torch.bmm(first_term, (self.f_grad*denom).unsqueeze(2)).squeeze(2)
        second_term = self.c**2 * (self.f_grad_square + self.diag_Sigma - self.v)
        return 0.5 * (first_term + second_term)

    def g(self, t, x):
        
        self.divide_input(x, t)

        v_reg = self.regularizer.regulariz_function(self.v)

        denom = 1/(torch.sqrt(v_reg) + self.eps)

        M_11 = torch.bmm(torch.diag_embed(denom), self.Sigma_sqrt)
        M_21 = -2*self.c*torch.bmm(torch.diag_embed(self.f_grad), self.Sigma_sqrt)
        M_22 = self.c *self.square_root_var_z_squared
        
        M_12 = torch.zeros_like(M_11)
        M_22 = torch.zeros_like(M_11)

        M_top = torch.cat((M_11, M_12), dim=2)  
        M_bottom = torch.cat((M_21, M_22), dim=2) 
        self.diffusion = torch.sqrt(self.tau) * torch.cat((M_top, M_bottom), dim=1)

        return self.diffusion

    
    
class RMSprop_SDE_1order_balistic_regime(RMSprop_SDE_2order_balistic_regime):
    def f(self, t, x):
        
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  

        # Theta coefficient
        v_reg = self.regularizer.regulariz_function(self.v)
        denom = 1/(torch.sqrt(v_reg) + self.eps)
        coef_theta = self.b_0_theta(denom)

        # V coefficient
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_v), dim = 1)

        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        return self.drift

    def g(self, t, x):
        
        self.divide_input(x, t)

        v_reg = self.regularizer.regulariz_function(self.v)
        denom = 1/(torch.sqrt(v_reg) + self.eps)
        M_11 = torch.bmm(torch.diag_embed(denom), self.Sigma_sqrt)
        M_21 = -2*self.c*torch.bmm(torch.diag_embed(self.f_grad), self.Sigma_sqrt)

        M_22 = torch.zeros_like(M_11)
        M_12 = torch.zeros_like(M_11)

        M_top = torch.cat((M_11, M_12), dim=2)  
        M_bottom = torch.cat((M_21, M_22), dim=2) 
        self.diffusion =  torch.sqrt(self.tau)*torch.cat((M_top, M_bottom), dim=1)
    
        return self.diffusion
    
class RMSprop_deterministic(RMSprop_SDE_2order_balistic_regime):
    def f(self, t, x):
        
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  

        # Theta coefficient
        v_reg = self.regularizer.regulariz_function(self.v)
        denom = 1/(torch.sqrt(v_reg) + self.eps)
        coef_theta = self.b_0_theta(denom)

        # V coefficient
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_v), dim = 1)

        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        
        return self.drift

    def g(self, t, x):
        
        self.diffusion = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)  
        return self.diffusion


def Discrete_RMProp_balistic_regime(funz, noise, tau, beta, c, num_steps, x_0, skip, epsilon = 1e-6, loss_bool = True, verbose = False):
    assert abs( (1 - beta) - tau * c) < 1e-6, "Check the parameters: 1 - beta should be equal to tau * c"
    
    batch_size = x_0.shape[0]
    path_x = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    path_v = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    if loss_bool:
        Loss_values = torch.zeros(batch_size, num_steps, device=x_0.device)
    path_x[:, 0, :] = x_0.detach().clone()
    path_v[:, 0, :] = torch.ones_like(x_0)

    max_lenghth_gamma_list = 1000
    noise_shuffled = noise[torch.randperm(noise.shape[0])]
    
    temp = 0
    temp1 = 0
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

        path_v[:, step+1] = (1 - c * tau) * v + c * tau * grad**2
        path_x[:, step+1] = x - tau * grad / (torch.sqrt(v) + epsilon)

        if (verbose and tau * step > temp) or True:
            temp += 1
            print(f'Step {step}, v: {v.mean().item():.4f}, theta: {x.mean().item():.4f} {path_x[:, step+1].mean().item():.4f}, grad: {grad.mean().item():.4f}, tau {tau}, epsilon {epsilon}')

    if loss_bool:
        Loss_values[:, -1] = funz.loss_batch(path_x[:, -1])

    if loss_bool:
        return torch.concat((path_x, path_v), dim=2), Loss_values
    else:
        return torch.concat((path_x, path_v), dim=2)