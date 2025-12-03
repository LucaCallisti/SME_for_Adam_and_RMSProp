import torch
import torchsde
import time
from Algorithms.Utils import Regularizer_ReLu, SDE_basic

torch.set_printoptions(precision=6)

class RMSprop_SDE_2order_batch_eq_regime(SDE_basic):
    def __init__(self, tau, c, function, All_time, regularizer, epsilon = 1e-6, sigma_value = 0.5, constant_noise = True, Verbose = True):
        super().__init__(noise_type="general")
        self.regime = 'batch_equivalent'
        self.eq = 'RMSProp'
        self.constant_noise = constant_noise
        
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
        self.divide_input(x, t)
        self.Verbose(t)       
        
        v_reg = self.regularizer.regulariz_function(self.v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(self.v)

        # Theta coefficient
        denom = 1/(torch.sqrt(v_reg) + self.eps)
        coef_theta = self.b_0_theta(denom) + self.tau * self.b_1_theta(denom, v_reg, v_reg_grad)

        # V coefficient
        coef_v = self.b_0_v() + self.tau * self.b_1_v(denom)

        self.drift = torch.concat((coef_theta, coef_v), dim = 1)
        self.is_it_Nan(self.drift, x, t, 'drift 2 order')

        return self.drift

    def b_0_theta(self, denom):
        return - self.f_grad  * denom
    def b_1_theta(self, denom, v_reg, v_reg_grad):
        OuterProduct = torch.einsum('ki,kj->kij', denom, denom)
        first_term_b_1_theta = torch.bmm(self.f_hessian*OuterProduct, self.f_grad.unsqueeze(2)).squeeze(2)+ self.c* 0.5* (denom**3) * (self.f_grad  * v_reg_grad) * (self.diag_Sigma - self.v)
        additional_term = 2 * torch.bmm( torch.bmm(torch.diag_embed(denom), self.term_b_1_theta_RMSProp_BatchEq), denom.unsqueeze(2)**2)

        return  - 0.5 * (first_term_b_1_theta + additional_term.squeeze(2))
    def b_0_v(self):
        return self.c * (self.diag_Sigma - self.v)
    def b_1_v(self, denom):
        first_term =  self.c * self.f_grad_square
        second_term = 0.5 * self.c**2 * (self.diag_Sigma  - self.v)
        if not self.constant_noise:
            additional_term = - 0.5 * torch.bmm( self.grad_Sigma_diag, self.f_grad  * denom)
            OuterProduct = torch.einsum('ki,kj->kij', denom, denom)
            sum = torch.einsum('bij, bjilk -> bkl', OuterProduct * self.Sigma, self.Sigma) 
            additional_term -= 0.5 * self.tau * sum 
        else:
            additional_term = 0

        return  first_term + second_term + additional_term

    def g(self, t, x):
        self.divide_input(x, t)

        v_reg = self.regularizer.regulariz_function(self.v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(self.v)
        
        denom = 1/(torch.sqrt(v_reg) + self.eps)

        M_11 = torch.bmm(torch.diag_embed(denom), self.Sigma_sqrt)
        OuterProduct = torch.einsum('ki,kj->kij',  denom, denom)
        Lambda_1 = 0.5 * torch.bmm( OuterProduct * self.f_hessian, self.Sigma_sqrt) + 0.25 * self.c * torch.bmm(torch.diag_embed( (denom**3)*(self.diag_Sigma - self.v) * v_reg_grad ), self.Sigma_sqrt)
        if not self.constant_noise:
            term_aux = torch.einsum('bi, bilk -> blk', denom * self.f_grad, self.grad_Sigma_sqrt)
            Lambda_1 -= 0.5 * torch.bmm(torch.diag_embed(denom) , term_aux)

            term_aux = torch.einsum('bij, bijlk -> blk', OuterProduct * self.Sigma, self.hessian_Sigma_sqrt)
            Lambda_1 += torch.bmm(torch.diag_embed(denom), term_aux)

            term_aux = torch.einsum('bij, bilk, bjlk -> blk', OuterProduct * self.Sigma, self.grad_Sigma_sqrt, self.grad_Sigma_sqrt)
            term_inv = torch.linalg.inv( torch.bmm(torch.diag_embed(denom), self.Sigma_sqrt))
            Lambda_1 += torch.bmm(term_inv , term_aux) 

        M_11 += self.tau * Lambda_1
        Lambda_2 = - 2*self.c*torch.bmm(torch.diag_embed(self.f_grad), self.Sigma_sqrt)
        if not self.constant_noise:
            Lambda_2 = Lambda_2 - self.c * 0.5 * torch.bmm( torch.bmm(self.grad_Sigma_diag, torch.diag_embed(denom)) , self.Sigma_sqrt )
        M_21 = self.tau * Lambda_2
        M_22 = self.c * torch.sqrt(self.tau)*self.square_root_var_z_squared
        
        M_12 = torch.zeros_like(M_11)

        M_top = torch.cat((M_11, M_12), dim=2)  
        M_bottom = torch.cat((M_21, M_22), dim=2)  
        self.diffusion =  torch.cat((M_top, M_bottom), dim=1)

        return self.diffusion
    
    def divide_input(self, x, t):
        aux = x.shape[1] // 2 
        theta, self.v = x[:, :aux], x[:, aux:2*aux]
        if (self.v<0).sum() > 0 and self.verbose: print('Warning: negative values in v', (self.v<0).sum().item(), self.v.min().item(), 'at time ', t)
        if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
        self.theta = theta
    
class RMSprop_SDE_1order_batch_eq_regime(RMSprop_SDE_2order_batch_eq_regime):
    def f(self, t, x):
        self.chronometer(t)  
        self.divide_input(x, t)
  
        self.Verbose(t)  
        if (self.v<0).any(): breakpoint()

        # Theta coefficient
        denom = 1/(torch.sqrt(self.v) + self.eps)
        coef_theta = self.b_0_theta(denom)

        # print(f'time: {t}, coef_theta norm: {coef_theta.norm()}, v norm: {self.v.norm()} sigma value: {self.sigma_value}')

        # V coefficient
        coef_v = self.b_0_v()
        self.drift = torch.concat((coef_theta, coef_v), dim = 1)

        self.is_it_Nan(self.drift, x, t, 'drift 1 order')
        
        return self.drift

    def g(self, t, x):
        self.divide_input(x, t)

        denom = 1/(torch.sqrt(self.v) + self.eps)
        M_11 = torch.bmm(torch.diag_embed(denom), self.Sigma_sqrt)
        M_21 = torch.zeros_like(M_11)
        M_22 = torch.zeros_like(M_11)
        M_12 = torch.zeros_like(M_11)

        M_top = torch.cat((M_11, M_12), dim=2)  
        M_bottom = torch.cat((M_21, M_22), dim=2) 
        self.diffusion = torch.cat((M_top, M_bottom), dim=1)

        return self.diffusion



def Discrete_RMProp_batch_eq_regime(funz, noise, tau, beta, c, num_steps, x_0, skip,  epsilon = 1e-6, verbose = False, loss_bool = True):
    assert abs( (1 - beta) - (tau) * c) < 1e-6, "Check the parameters: 1 - beta should be equal to tau * c"

    batch_size = x_0.shape[0]
    path_x = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    path_v = torch.zeros(batch_size, num_steps, x_0.shape[1], device=x_0.device)
    if loss_bool:
        Loss_values = torch.zeros(batch_size, num_steps, device=x_0.device)
    path_x[:, 0, :] = x_0.detach().clone()
    path_v[:, 0, :] = torch.ones_like(x_0)

    tau = torch.tensor(tau, device=x_0.device)
    temp = 0
    max_lenghth_gamma_list = 1000
    noise_shuffled = noise[torch.randperm(noise.shape[0])]
    
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

        grad = funz.noisy_grad_batcheq(x, gamma, tau)

        path_v[:, step+1] = beta * v + tau**2 * c * grad**2
        path_x[:, step+1] = x - tau * grad / (torch.sqrt(v) + epsilon)

        # path_v[:, step+1] = beta * v + lr**2 * c * torch.pow(expected_grad, 2) + lr * c * noise**2 + 2 * lr**(3/2) * c  * noise * expected_grad
        # path_x[:, step+1] = x - lr * expected_grad / (torch.sqrt(path_v[:, step]) + epsilon) - torch.sqrt(lr) * noise / (torch.sqrt(path_v[:, step]) + epsilon) 

        if verbose and step*tau >= temp:
            temp += 1
            print(f'Step {step}, v: {path_v[step]}, theta: {path_x[step]}')
    if loss_bool:
        Loss_values[:, -1] = funz.loss_batch(path_x[:, -1])
        return torch.concat((path_x, path_v), dim = 2), Loss_values
    else:
        return torch.concat((path_x, path_v), dim = 2)