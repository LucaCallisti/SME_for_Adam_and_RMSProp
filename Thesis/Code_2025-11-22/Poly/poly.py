import math
import torch




class base_poly:
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
    def term_b1_RMSProp_BatchEq(self, x):
        return (self.Diag_sigma(x) * self.f_third(x)).unsqueeze(-1).unsqueeze(-1)


class aux_function_poly_Wshaped:
    def __init__(self, y1, y2, constant_term, multiplicative_factor = 1):
        self.y1 = torch.tensor(y1)
        self.y2 = torch.tensor(y2)
        self.constant_term = torch.tensor(constant_term)
        self.multiplicative_factor = torch.tensor(multiplicative_factor)

    def f(self, x):
        return self.multiplicative_factor * (x - self.y1)**2 * (x - self.y2)**2 + self.constant_term

    def f_prime(self, x):
        term1 = x - self.y1
        term2 = x - self.y2
        return 2 * self.multiplicative_factor * ( term1**2 * term2 + term2**2 * term1 )

    def f_second(self, x):
        term1 = x - self.y1
        term2 = x - self.y2
        return 2 * self.multiplicative_factor * ( term1**2 + term2**2 + 4 * term1 * term2 )
    
    def f_third(self, x):
        term1 = x - self.y1
        term2 = x - self.y2
        return 12 * self.multiplicative_factor * ( term1 + term2 )        
    

class function_poly_Wshaped(base_poly):
    def __init__(self, x1, x2, delta, m = 1):
        '''
        The function is defined as: f_1 (x) = m*(x- (x1-delta) )^2 * (x- (x1-delta) )^2,  f_2 (x) = m*(x- (x1+delta) )^2 * (x- (x2+delta) )^2
        and f(x) = 0.5 * m * ( f_1 (x) + f_2 (x) )
        '''
        assert x1**2 + x2**2 > 4*(x1*x2 + 3*delta**2), "The chosen parameters do not ensure the presence of four minimas."
        self.coeff_1 = (x1 - delta, x2 - delta, 0) 
        self.coeff_2 = (x1 + delta, x2 + delta, 0)
        sqrt_value = math.sqrt( (x1 + x2)**2 - 4*(x1*x2 + 3*delta**2) )
        self.coeff = (0.5 * (x1 + x2 - sqrt_value), 0.5 * (x1 + x2 + sqrt_value), delta**2 * ( (x1 + x2)**2 -6*x1*x2 ) - 8 * delta**4)

        self._f = aux_function_poly_Wshaped( self.coeff[0], self.coeff[1], self.coeff[2], multiplicative_factor = m )
        self._f1 = aux_function_poly_Wshaped( self.coeff_1[0], self.coeff_1[1], self.coeff_1[2], multiplicative_factor = m )
        self._f2 = aux_function_poly_Wshaped( self.coeff_2[0], self.coeff_2[1], self.coeff_2[2], multiplicative_factor = m )

    def f(self, x):
        return self._f.f(x)
    def f_prime(self, x):
        return self._f.f_prime(x)
    def f_second(self, x):
        return self._f.f_second(x)
    def f_third(self, x):
        return self._f.f_third(x)
    def f1(self, x):
        return self._f1.f(x)
    def f1_prime(self, x):
        return self._f1.f_prime(x)
    def f1_second(self, x):
        return self._f1.f_second(x)
    def f2(self, x):
        return self._f2.f(x)
    def f2_prime(self, x):
        return self._f2.f_prime(x)
    def f2_second(self, x):
        return self._f2.f_second(x)
    def f1_third(self, x):
        return self._f1.f_third(x)
    def f2_third(self, x):
        return self._f2.f_third(x)



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