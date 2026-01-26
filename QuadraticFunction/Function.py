import torch
import QuadraticFunction.utils as utils

def create_def_positive_matrix(dim):
    """
    Creates a random symmetric positive definite matrix of given dimension.
    Args:
        dim: Dimension of the matrix.
    Returns:
        A symmetric positive definite matrix as a torch tensor.
    """
    A = torch.randn(dim, dim)
    A = A @ A.T  
    A += torch.eye(dim) * 0.1
    return A

class Quadratic_function():
    """
    Represents a quadratic function of the form:
        f(x, gamma) = 0.5 * (x - gamma)^T A (x - gamma) - 0.5 * tr(A)
    where gamma is a standard normal vector.
    Provides methods for function evaluation, gradient, Hessian, and related operations.
    """
    def __init__(self, dim=10, std = 1, A=None, dataset = None):
        """
        Initializes the quadratic function with a random or provided matrix A.
        Args:
            dim: Dimension of the function.
            std: Standard deviation for noise.
            A: Optional matrix to use instead of random.
            dataset: Optional dataset (unused).
        """
        if A is not None:
            self.A = A
        else:
            self.A = create_def_positive_matrix(dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.A = self.A.to(device)
        self.sigma = std**2 * self.A @ self.A.T
        self.sigma_sqrt = std * self.A
        self.sigma = std**2 * self.A @ self.A.T
        self.sigma_sqrt = std * self.A
        sigma_sigma_T_squared = self.sigma**2
        self.M_matrix = 2 * sigma_sigma_T_squared 

        eigenvalues, eigenvectors = torch.linalg.eigh(self.M_matrix)
        self.sqrt_M_matrix = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T


    def change_batch_size(self, batch_size):
        """
        Expands internal matrices to match the given batch size for batched operations.
        Args:
            batch_size: The batch size to expand to.
        """
        self.A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1).to(self.A.device)
        self.sigma_expanded = self.sigma.unsqueeze(0).expand(batch_size, -1, -1).to(self.A.device)
        self.sqrt_M_matrix_expanded = self.sqrt_M_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(self.A.device)
        self.daig_sigma_expanded = torch.diag(self.sigma).unsqueeze(0).expand(batch_size, -1, -1).squeeze(1).to(self.A.device)
        self.sigma_sqrt_expanded = self.sigma_sqrt.unsqueeze(0).expand(batch_size, -1, -1).to(self.A.device)
        
    def function(self, x, gamma):
        """
        Evaluates the quadratic function for input x and gamma.
        Args:
            x: Input tensor.
            gamma: Random vector tensor.
        Returns:
            Function value as a tensor.
        """
        batch_size = x.shape[0]
        diff = x - gamma  
        diff_unsqueezed = diff.unsqueeze(1)

        result = torch.bmm(diff_unsqueezed, torch.bmm(self.A_expanded, diff_unsqueezed.transpose(1, 2)))
        result = result.squeeze()  
        
        return 0.5 * result - 0.5 * torch.trace(self.A) * torch.ones(batch_size, device=x.device)
    
    def expected_value(self, x):        
        """
        Computes the expected value of the quadratic function for input x.
        Args:
            x: Input tensor.
        Returns:
            Expected value as a tensor.
        """
        x_unsqueezed = x.unsqueeze(1)
        
        result = torch.bmm(x_unsqueezed, torch.bmm(self.A_expanded, x_unsqueezed.transpose(1, 2)))
        result = result.squeeze()  
        
        return 0.5 * result
    
    def gradient(self, x, gamma):
        """
        Computes the gradient of the quadratic function with respect to x.
        Args:
            x: Input tensor.
            gamma: Random vector tensor.
        Returns:
            Gradient tensor.
        """
        diff = x - gamma   
        result = torch.bmm(self.A_expanded, diff.unsqueeze(2))
        return result.squeeze(2) 
    
    def expected_value_gradient(self, x):
        """
        Computes the gradient of the expected value of the quadratic function.
        Args:
            x: Input tensor.
        Returns:
            Gradient tensor.
        """
        return self.gradient(x, torch.zeros_like(x))
    def hessian(self, x):
        """
        Returns the Hessian matrix (constant for quadratic functions).
        Args:
            x: Input tensor.
        Returns:
            Hessian matrix tensor.
        """
        return self.A_expanded  
    def Sigma(self, x):
        """
        Returns the sigma matrix for the quadratic function.
        Args:
            x: Input tensor.
        Returns:
            Sigma matrix tensor.
        """
        return self.sigma_expanded
    def Sigma_sqrt(self, x):
        """
        Returns the square root of the sigma matrix.
        Args:
            x: Input tensor.
        Returns:
            Square root of sigma matrix tensor.
        """
        return self.sigma_sqrt_expanded
    def Diag_sigma(self, x):
        """
        Returns the diagonal of the sigma matrix.
        Args:
            x: Input tensor.
        Returns:
            Diagonal of sigma matrix tensor.
        """
        return self.daig_sigma_expanded
    def square_root_var_z_squared(self, x):
        """
        Returns the square root of the variance of z squared for the quadratic function.
        Args:
            x: Input tensor.
        Returns:
            Square root variance tensor.
        """
        return self.sqrt_M_matrix_expanded
    def noisy_grad_balistic(self, x, gamma):
        """
        Returns the noisy gradient for the balistic regime.
        Args:
            x: Input tensor.
            gamma: Random vector tensor.
        Returns:
            Noisy gradient tensor.
        """
        return self.gradient(x, gamma)
    def noisy_grad_batcheq(self, x, gamma, tau):
        """
        Returns the noisy gradient for the batch equivalent regime.
        Args:
            x: Input tensor.
            gamma: Random vector tensor.
            tau: Scaling parameter.
        Returns:
            Noisy gradient tensor.
        """
        return self.gradient(x, gamma / tau**0.5)
    def grad(self, x):
        """
        Returns the gradient of the quadratic function at x.
        Args:
            x: Input tensor.
        Returns:
            Gradient tensor.
        """
        return self.gradient(x, torch.zeros_like(x))
    def term_b1_RMSProp_BatchEq(self, x, denom):
        """
        Returns the term b1 for RMSProp in the batch equivalent regime (currently zeros).
        Args:
            x: Input tensor.
            denom: Denominator tensor (unused).
        Returns:
            Zero tensor of same shape as x.
        """
        return torch.zeros_like(x)

def create_dataset_and_initial_point(dim, num_samlpes, path=None):
    """
    Creates a dataset of random samples and an initial point for experiments.
    Args:
        dim: Dimension of the data.
        num_samlpes: Number of samples to generate.
        path: Optional path to save the dataset.
    Returns:
        Tuple of (dataset, initial_point).
    """
    utils.set_seed(0)
    initial_point = 10*torch.randn(dim)
    dataset = torch.randn(num_samlpes, dim)
    if path is not None:
        torch.save({'dataset': dataset, 'initial_point': initial_point}, path)
    return dataset, initial_point
