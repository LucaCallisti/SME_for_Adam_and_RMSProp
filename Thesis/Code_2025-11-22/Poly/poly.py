import math
import torch
from torch.func import grad, vmap


class base_poly:
    '''
    Base class for functions defined as f(x) = 1/2 * ( f_1 (x) + f_2 (x) ).
    '''
    def noisy_grad_balistic(self, x, gamma):
        f1_vals = self.f1_prime(x)
        f2_vals = self.f2_prime(x)
        if gamma.ndim == 3:
            batch_size = gamma.shape[2] 
            f1_vals = f1_vals.unsqueeze(2).expand(-1, -1, batch_size)
            f2_vals = f2_vals.unsqueeze(2).expand(-1, -1, batch_size)
            grad = torch.where(gamma == 0, f1_vals, f2_vals)
            grad = grad.mean(dim = 2)
        else:
            grad = torch.where(gamma == 0, f1_vals, f2_vals)
        return grad
    def noisy_grad_batcheq(self, x, gamma, tau):
        return self.noisy_grad_balistic(x, gamma)

    def grad(self, x):
        return self.f_prime(x)
    def hessian(self, x):
        return self.f_second(x).unsqueeze(-1)
    def Diag_sigma(self, x):
        return 0.25 * (self.f1_prime(x) - self.f2_prime(x))**2
    def Sigma(self, x):
        return self.Diag_sigma(x).unsqueeze(-1)
    def Sigma_sqrt(self, x):
        return torch.sqrt( self.Diag_sigma(x).clamp(min=1e-12) ).unsqueeze(-1)
    def square_root_var_z_squared(self, x):
        return (0.75 * (self.f1_prime(x) - self.f2_prime(x))**2).unsqueeze(-1)
    def grad_sigma(self, x):
        return ( 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_second(x) - self.f2_second(x)) ).unsqueeze(-1).unsqueeze(-1)
    def grad_sigma_diag(self, x):
        return ( 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_second(x) - self.f2_second(x)) ).unsqueeze(-1)
    def hessian_sigma(self, x):
        return ( 0.5 * (self.f1_second(x) - self.f2_second(x))**2 + 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_third(x) - self.f2_third(x)) ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    def term_b1_RMSProp_BatchEq(self, x, denom):
        return (self.Diag_sigma(x) * self.f_third(x) * denom**3).unsqueeze(-1)
    def grad_sigma_sqrt(self, x):
        return 0.5 * ( self.grad_sigma(x) / self.Sigma_sqrt(x).unsqueeze(-1) )
    def hessian_sigma_sqrt(self, x):
        term1 = self.hessian_sigma(x) / self.Sigma_sqrt(x).unsqueeze(-1).unsqueeze(-1)
        term2 = self.grad_sigma_sqrt(x).unsqueeze(-1) * self.grad_sigma(x).unsqueeze(-1) / self.Sigma(x).unsqueeze(-1).unsqueeze(-1)
        return 0.5 * ( term1 + term2 )
    def hessian_sigma_diag(self, x):
        return self.hessian_sigma(x).squeeze(-1)




class Poly2(base_poly):
    def __init__(self, x1, x2, c, d):
        '''
        The function is defined as: f_1 (x) = 2 * c*(x- x1 )^2 * (x- x2 )^2,  f_2 (x) =   2 * d * x^2
        and f(x) = 0.5 * ( f_1 (x) + f_2 (x) )
        '''

        self.x_liminf = -1.5
        self.x_limsup = 3

        self._f1 = lambda x: 2 * c * (x-x1)**2 * (x - x2)**4
        self._f1_prime = lambda x: 2 * c * ( 2*(x - x1) * (x - x2)**4 + 4*(x - x2)**3 * (x - x1)**2 )
        self._f1_second = lambda x: 2 * c * ( 2*(x - x2)**4 + 8*(x - x1)*(x - x2)**3 + 12*(x - x2)**2 * (x - x1)**2 + 8*(x - x2)**3 * (x - x1) )
        self._f1_third = lambda x: 2 * c * ( 8*(x - x2)**3 + 8*(x - x2) + 24*(x - x1)*(x - x2)**2 + 24*(x - x2)*(x - x1)**2 + 24*(x - x1)*(x - x2)**2 + 8*(x - x2)**3 + 24*(x - x2)**2 * (x - x1) )

        self._f2 = lambda x: 2 * d * x**2
        self._f2_prime = lambda x: 4 * d * x
        self._f2_second = lambda x: 4 * d * torch.ones_like(x)
        self._f2_third = lambda x: torch.zeros_like(x)

        self._f = lambda x: 0.5 * ( self._f1(x) + self._f2(x) )
        self._f_prime = lambda x: 0.5 * ( self._f1_prime(x) + self._f2_prime(x) )
        self._f_second = lambda x: 0.5 * ( self._f1_second(x) + self._f2_second(x) )
        self._f_third = lambda x: 0.5 * ( self._f1_third(x) + self._f2_third(x) )
        
    def f(self, x):
        return self._f(x)
    def f_prime(self, x):
        return self._f_prime(x)
    def f_second(self, x):
        return self._f_second(x)
    def f_third(self, x):
        return self._f_third(x)
    def f1(self, x):
        return self._f1(x)
    def f1_prime(self, x):
        return self._f1_prime(x)
    def f1_second(self, x):
        return self._f1_second(x)
    def f1_third(self, x):
        return self._f1_third(x)
    def f2(self, x):
        return self._f2(x)
    def f2_prime(self, x):
        return self._f2_prime(x)
    def f2_second(self, x):
        return self._f2_second(x)
    def f2_third(self, x):
        return self._f2_third(x)
    

class Poly_with_additional_noise(base_poly):
    def __init__(self, x1, x2, c, d, noise_level=0):
        '''
        The function is defined as: f_1 (x) = 2 * c*(x- x1 )^2 * (x- x2 )^4,  f_2 (x) =   2 * d * x^2
        and f(x) = 0.5 * ( f_1 (x) + f_2 (x) )
        '''

        self.x_liminf = -1.5
        self.x_limsup = 3

        # Translation of the original functions
        translation_quantitiy = 0.1
        x1 = x1 - translation_quantitiy
        x2 = x2 - translation_quantitiy

        self.f1_old = lambda x: 2 * c * (x-x1)**2 * (x - x2)**4
        self.f1_prime_old = lambda x: 2 * c * ( 2*(x - x1) * (x - x2)**4 + 4*(x - x2)**3 * (x - x1)**2 )
        self.f1_second_old = lambda x: 2 * c * ( 2*(x - x2)**4 + 8*(x - x1)*(x - x2)**3 + 12*(x - x2)**2 * (x - x1)**2 + 8*(x - x2)**3 * (x - x1) )
        self.f1_third_old =  lambda x: 2 * c * ( 8*(x - x2)**3 + 8*(x - x2) + 24*(x - x1)*(x - x2)**2 + 24*(x - x2)*(x - x1)**2 + 24*(x - x1)*(x - x2)**2 + 8*(x - x2)**3 + 24*(x - x2)**2 * (x - x1) )

        # Traslation of the original functions
        self.f2_old = lambda x: 2 * d * (x - translation_quantitiy)**2
        self.f2_prime_old = lambda x: 4 * d * (x - translation_quantitiy)
        self.f2_second_old = lambda x: 4 * d * torch.ones_like(x)
        self.f2_third_old = lambda x: torch.zeros_like(x)

        if noise_level == 0:
            self._f1 = self.f1_old
            self._f1_prime = self.f1_prime_old
            self._f1_second = self.f1_second_old
            self._f1_third = self.f1_third_old
            self._f2 = self.f2_old
            self._f2_prime = self.f2_prime_old
            self._f2_second = self.f2_second_old
            self._f2_third = self.f2_third_old
        else:
            cost = noise_level
            self.g = lambda x : cost * (self.f1_old(x) - self.f2_old(x))
            self.g_prime = lambda x : cost * (self.f1_prime_old(x) - self.f2_prime_old(x))
            self.g_second = lambda x : cost * (self.f1_second_old(x) - self.f2_second_old(x))
            self.g_third = lambda x : cost * (self.f1_third_old(x) - self.f2_third_old(x))

            self._f1 = lambda x : self.f1_old(x) + self.g(x)
            self._f1_prime = lambda x : self.f1_prime_old(x) + self.g_prime(x)
            self._f1_second = lambda x : self.f1_second_old(x) + self.g_second(x)
            self._f1_third = lambda x : self.f1_third_old(x) + self.g_third(x)
            self._f2 = lambda x : self.f2_old(x) - self.g(x)
            self._f2_prime = lambda x : self.f2_prime_old(x) - self.g_prime(x)
            self._f2_second = lambda x : self.f2_second_old(x) - self.g_second(x)
            self._f2_third = lambda x : self.f2_third_old(x) - self.g_third(x)

        self._f = lambda x: 0.5 * ( self._f1(x) + self._f2(x) )
        self._f_prime = lambda x: 0.5 * ( self._f1_prime(x) + self._f2_prime(x) )
        self._f_second = lambda x: 0.5 * ( self._f1_second(x) + self._f2_second(x) )
        self._f_third = lambda x: 0.5 * ( self._f1_third(x) + self._f2_third(x) )
        
    def f(self, x):
        return self._f(x)
    def f_prime(self, x):
        return self._f_prime(x)
    def f_second(self, x):
        return self._f_second(x)
    def f_third(self, x):
        return self._f_third(x)
    def f1(self, x):
        return self._f1(x)
    def f1_prime(self, x):
        return self._f1_prime(x)
    def f1_second(self, x):
        return self._f1_second(x)
    def f1_third(self, x):
        return self._f1_third(x)
    def f2(self, x):
        return self._f2(x)
    def f2_prime(self, x):
        return self._f2_prime(x)
    def f2_second(self, x):
        return self._f2_second(x)
    def f2_third(self, x):
        return self._f2_third(x)