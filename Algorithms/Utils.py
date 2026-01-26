import torch
import torchsde
import time
from typing import Dict, Any

class Regularizer_ReLu():
    """
    Implements a ReLU-based regularizer for optimization algorithms.
    """
    def set_costant(self, cost):
        """
        Set the minimum value for the regularizer.
        Args:
            cost: Minimum value tensor.
        """
        self.u_min = cost.clone()
    def regulariz_function(self, u):
        """
        Apply the regularization function (ReLU variant).
        Args:
            u: Input tensor.
        Returns:
            Regularized tensor.
        """
        self.u_min = self.u_min.to(u.device)
        return torch.max(u, self.u_min)
    def derivative_regulariz_function(self, u):
        """
        Compute the derivative of the regularization function.
        Args:
            u: Input tensor.
        Returns:
            Derivative tensor.
        """
        return torch.where(u > self.u_min, torch.ones_like(u), torch.zeros_like(u))
    

class SDE_basic(torchsde.SDEIto):
    """
    Base class for SDE optimizers, providing shared utilities and interface.
    """
    def __init__(self, noise_type="general"):
        """
        Initialize the base SDE class.
        Args:
            noise_type: Type of noise for the SDE.
        """
        super().__init__(noise_type=noise_type)
        self.eq = None
        self.constant_noise = None
        self.last_neg_value = 0
        
    def update_quantities(self, theta, t):
        """
        Update all relevant quantities (gradients, Hessians, etc.) for the SDE.
        Args:
            theta: Current parameter tensor.
            t: Current time.
        """
        self.diffusion = None
        self.drift = None
 
        self.f_grad = self.fun.grad(theta)
        assert self.f_grad.dim() == 2, "f_grad should be of shape (batch_size, dim)"
        self.f_hessian = self.fun.hessian(theta)
        assert self.f_hessian.dim() == 3, "f_hessian should be of shape (batch_size, dim, dim)"
        self.Sigma_sqrt = self.sigma_value*self.fun.Sigma_sqrt(theta)
        assert self.Sigma_sqrt.dim() == 3, "Sigma_sqrt should be of shape (batch_size, dim, dim)"
        self.diag_Sigma = self.sigma_value**2 * self.fun.Diag_sigma(theta)
        assert self.diag_Sigma.dim() == 2, "diag_Sigma should be of shape (batch_size, dim)"
        self.square_root_var_z_squared = self.sigma_value**2 * self.fun.square_root_var_z_squared(theta)
        assert self.square_root_var_z_squared.dim() == 3, "square_root_var_z_squared should be of shape (batch_size, dim, dim)"
        self.f_grad_square = torch.pow(self.f_grad, 2)   

        if self.eq == 'RMSProp' and self.regime == 'batch_equivalent':
            v_reg = self.regularizer.regulariz_function(self.v)
            denom = 1/(torch.sqrt(v_reg) + self.eps)
            term_RMSProp_BatchEq = self.fun.term_b1_RMSProp_BatchEq(theta, denom)
            assert term_RMSProp_BatchEq.dim() == 2, "term_RMSProp_BatchEq should be of shape (batch_size, dim)"
            self.term_b_1_theta_RMSProp_BatchEq = self.sigma_value**2 * term_RMSProp_BatchEq * denom

        if self.constant_noise is False:
            self.Sigma = self.sigma_value **2 * self.fun.Sigma(theta)
            assert self.Sigma.dim() == 3, "Sigma should be of shape (batch_size, dim, dim)"
            self.grad_Sigma = self.sigma_value**2 * self.fun.grad_sigma(theta)
            assert self.grad_Sigma.dim() == 4, "grad_Sigma should be of shape (batch_size, dim, dim, dim)"  # where the first dim is for the derivative
            self.grad_Sigma_diag = self.sigma_value**2 * self.fun.grad_sigma_diag(theta)
            assert self.grad_Sigma_diag.dim() == 3, "grad_Sigma_diag should be of shape (batch_size, dim, dim)"  # where the first dim is for the derivative
            self.hessian_Sigma = self.sigma_value**2 * self.fun.hessian_sigma(theta)
            assert self.hessian_Sigma.dim() == 5, "hessian_Sigma should be of shape (batch_size, dim, dim, dim, dim)"  # where the first and second dim is for the second derivative
            self.hessian_Sigma_diag = self.sigma_value**2 * self.fun.hessian_sigma_diag(theta)
            assert self.hessian_Sigma_diag.dim() == 4, "hessian_Sigma_diag should be of shape (batch_size, dim, dim, dim)"  # where the first and second dim is for the second derivative
            self.grad_Sigma_sqrt = self.sigma_value * self.fun.grad_sigma_sqrt(theta)
            assert self.grad_Sigma_sqrt.dim() == 4, "grad_Sigma_sqrt should be of shape (batch_size, dim, dim, dim)"  # where the first dim is for the derivative
            self.hessian_Sigma_sqrt = self.sigma_value * self.fun.hessian_sigma_sqrt(theta)
            assert self.hessian_Sigma_sqrt.dim() == 5, "hessian_Sigma_sqrt should be of shape (batch_size, dim, dim, dim, dim)"  # where the first and second dim is for the second derivative



    def is_it_Nan(self, input, x, t, where):
        """
        Check for NaN values in the input tensor and print a warning if found.
        Args:
            input: Tensor to check.
            x: State tensor.
            t: Current time.
            where: Description string for context.
        Returns:
            False (for compatibility).
        """
        if torch.isnan(input).any() and t > self.t_nan:
            print(f"Warning: {torch.isnan(input).sum()} NaN values detected in {where}")
            self.t_nan = t + 1
        return False
    def chronometer(self, t):
        """
        Print timing information for function calls (for verbose/debugging).
        Args:
            t: Current time.
        """
        if self.i == 1: self.start_new_f = time.time()
        if self.i > 1:      
            print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t:.4f}')
            self.start_new_f = time.time()
        self.i += 1
    def Verbose(self, t):
        """
        Print verbose information if enabled and at the right time interval.
        Args:
            t: Current time.
        """

        if (self.verbose and t > self.t_verbose): 
            self.chronometer(t)
            print(f't: {t:.1f}', end='\r')
            self.t_verbose = t + 1
   
    def divide_input(self, x, t):
        """
        Divide the input state x into optimizer-specific components (theta, m, v, etc.).
        Args:
            x: State tensor.
            t: Current time.
        """
        if self.eq == 'RMSProp':
            aux = x.shape[1] // 2 
            theta, self.v = x[:, :aux], x[:, aux:2*aux]
            if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
            self.theta = theta
        elif self.eq == 'Adam':
            aux = x.shape[1] // 3
            theta, self.m, self.v = x[:, :aux], x[:, aux:2*aux], x[:, 2*aux:]
            if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
            self.theta = theta
        else:
            raise ValueError(f"Unknown equation type: {self.eq}")
        if (self.v<0).sum() > 0 and self.verbose and t > self.last_neg_value + 1: 
            print('Warning: negative values in v', (self.v<0).sum().item(), self.v.min().item(), 'at time ', t)
            self.last_neg_value = t

def get_regime_functions(regime: str, optimizer: str) -> Dict[str, Any]:
    """
    Import and return the appropriate functions for the selected regime and optimizer.
    Args:
        regime: Either 'balistic' or 'batch equivalent'.
        optimizer: The optimizer to use ('Adam' or 'RMSProp').
    Returns:
        Dictionary containing regime-specific functions and classes.
    """

    if regime == 'balistic':
        if optimizer == 'Adam':
            from Algorithms.Balistic_regime_Adam import (
                Discrete_Adam_balistic_regime,
                Adam_SDE_2order_balistic_regime,
                Adam_deterministic,
                Adam_SDE_1order_balistic_regime
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_Adam_balistic_regime,
                'approx_1_fun_det': Adam_deterministic,
                'approx_2_fun': Adam_SDE_2order_balistic_regime,
                'approx_1_fun': Adam_SDE_1order_balistic_regime
            }
        elif optimizer == 'RMSProp':
            from Algorithms.Balistic_regime_RMSProp import (
                Discrete_RMProp_balistic_regime,
                RMSprop_SDE_2order_balistic_regime,
                RMSprop_deterministic,
                RMSprop_SDE_1order_balistic_regime
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_RMProp_balistic_regime,
                'approx_1_fun_det': RMSprop_deterministic,
                'approx_2_fun': RMSprop_SDE_2order_balistic_regime,
                'approx_1_fun': RMSprop_SDE_1order_balistic_regime
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    elif regime == 'batch_equivalent':
        if optimizer == 'Adam':
            from Algorithms.Batch_eq_regime_Adam import (
                Discrete_Adam_batch_equivalent_regime,
                Adam_SDE_1order_batch_equivalent_regime,
                Adam_SDE_2order_batch_equivalent_regime,
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_Adam_batch_equivalent_regime,
                'approx_1_fun': Adam_SDE_1order_batch_equivalent_regime,
                'approx_2_fun': Adam_SDE_2order_batch_equivalent_regime
            }
        elif optimizer == 'RMSProp':
            from Algorithms.Batch_eq_regime_RMSProp import (
                Discrete_RMProp_batch_eq_regime,
                RMSprop_SDE_1order_batch_eq_regime,
                RMSprop_SDE_2order_batch_eq_regime,
            )
            return {
                'regularizer': Regularizer_ReLu(),
                'discr_fun': Discrete_RMProp_batch_eq_regime,
                'approx_1_fun': RMSprop_SDE_1order_batch_eq_regime,
                'approx_2_fun': RMSprop_SDE_2order_batch_eq_regime
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    else:
        raise ValueError(f"Unknown regime: {regime}")
    

def get_batch_size(sigma, tau, regime):
    """
    Compute the batch size given sigma, tau, and regime.
    Args:
        sigma: Noise scale.
        tau: Step size.
        regime: Regime string.
    Returns:
        Batch size (int).
    """
    if regime == 'balistic':
        b_size = int( sigma**(-2) / 1 )
    elif regime == 'batch_equivalent':
        b_size = int( tau * sigma**(-2) )
    else:
        raise ValueError(f"Unknown regime: {regime}")
    assert b_size > 0, "Batch size must be positive"
    return b_size
def get_sigma(batch_size, tau, regime):
    """
    Compute the sigma value given batch size, tau, and regime.
    Args:
        batch_size: Batch size.
        tau: Step size.
        regime: Regime string.
    Returns:
        Sigma value (float).
    """
    if regime == 'balistic':
        return (1 * batch_size)**(-0.5)
    elif regime == 'batch_equivalent':
        return (tau / batch_size)**(0.5)
    else:
        raise ValueError(f"Unknown regime: {regime}")
