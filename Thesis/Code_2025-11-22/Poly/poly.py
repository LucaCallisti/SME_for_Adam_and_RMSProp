import sympy as sp
import math
import torch

class function:
    def __init__(self, m1, m2, constant_term):
        self.m1 = torch.tensor(m1)
        self.m2 = torch.tensor(m2)
        self.constant_term = torch.tensor(constant_term)

    def f(self, x):
        return (x - self.m1)**2 * (x - self.m2)**2 + self.constant_term

    def f_prime(self, x):
        sqrt_f = torch.sqrt(self.f(x))
        return 2 * (x - self.m1) * sqrt_f + 2 * (x - self.m2) * sqrt_f
    
    def f_second(self, x):
        term1 = x - self.m1
        term2 = x - self.m2
        sum = term1 + term2
        return 2 * ( sum**2 + term1 * term2 )
    
    def f_third(self, x):
        term1 = x - self.m1
        term2 = x - self.m2
        return 8 * (term1 + term2) + 2 * term1 + 2 * term2        
    

class function_poly:
    def __init__(self, x1, x2, delta):
        '''
        The function is defined as: f_1 (x) = (x- (x1-delta) )^2 * (x- (x1-delta) )^2,  f_2 (x) = (x- (x1+delta) )^2 * (x- (x2+delta) )^2
        and f(x) = 0.5 ( f_1 (x) + f_2 (x) )
        '''
        assert x1**2 + x2**2 > 4*(x1*x2 + 3*delta**2), "The chosen parameters do not ensure the presence of four minimas."
        self.coeff_1 = (x1 - delta, x2 - delta, 0) 
        self.coeff_2 = (x1 + delta, x2 + delta, 0)
        sqrt_value = math.sqrt( (x1 + x2)**2 - 4*(x1*x2 + 3*delta**2) )
        self.coeff = (0.5 * (x1 + x2 - sqrt_value), 0.5 * (x1 + x2 + sqrt_value), delta**2 * ( (x1 + x2)**2 -6*x1*x2 ) - 8 * delta**4)

        self.f = function( self.coeff[0], self.coeff[1], self.coeff[2] )
        self.f1 = function( self.coeff_1[0], self.coeff_1[1], self.coeff_1[2] )
        self.f2 = function( self.coeff_2[0], self.coeff_2[1], self.coeff_2[2] )

    def f(self, x):
        return self.f.f(x)
    def f_prime(self, x):
        return self.f.f_prime(x)
    def f_second(self, x):
        return self.f.f_second(x)
    def f1(self, x):
        return self.f1.f(x)
    def f1_prime(self, x):
        return self.f1.f_prime(x)
    def f1_second(self, x):
        return self.f1.f_second(x)
    def f2(self, x):
        return self.f2.f(x)
    def f2_prime(self, x):
        return self.f2.f_prime(x)
    def f2_second(self, x):
        return self.f2.f_second(x)
    def f1_third(self, x):
        return self.f1.f_third(x)
    def f2_third(self, x):
        return self.f2.f_third(x)
    
    def grad(self, x, gamma):
        f1_vals = self.f1_prime(x)
        f2_vals = self.f2_prime(x)
        return torch.where(gamma == 0, f1_vals, f2_vals)
    
    def autograd_vector_batch(self, x):
        return self.f_prime(x)
    def hessian_matrix_batch(self, x):
        return self.f_second(x).unsqueeze(-1)
    def Diag_sigma(self, x):
        return 0.25 * (self.f1_prime(x)**2 - self.f2_prime(x)**2)
    def Sigma_sqrt(self, x):
        return torch.sqrt( self.Diag_sigma(x).clamp(min=1e-12) ).unsqueeze(-1)
    def square_root_var_z_squared(self, x):
        return torch.zeros_like(x).unsqueeze(-1)
    def grad_sigma(self, x):
        return ( 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_second(x) - self.f2_second(x)) ).unsqueeze(-1).unsqueeze(-1)
    def grad_sigma_diag(self, x):
        return ( 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_second(x) - self.f2_second(x)) ).unsqueeze(-1)
    def hessian_sigma(self, x):
        return ( 0.5 * (self.f1_second(x) - self.f2_second(x))**2 + 0.5 * (self.f1_prime(x) - self.f2_prime(x)) * (self.f1_third(x) - self.f2_third(x)) ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
