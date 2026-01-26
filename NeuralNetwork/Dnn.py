import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from NeuralNetwork.Utils import load_and_preprocess_data


class base_nn:
    """
    Base class for neural network models with batched parameter support and utility methods.
    """
    def forward_batch(self, x: torch.Tensor, theta_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply the network to a batch of parameter vectors using torch.func.
        Args:
            x: Fixed input tensor.
            theta_batch: Batch of parameter vectors (B, total_params).
        Returns:
            Output tensor of shape (B, output_dim).
        """

        def forward_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            return torch.func.functional_call(self.network, params_dict, (x,))

        batched_forward = F.vmap(forward_single)
        return batched_forward(theta_batch)

    def get_loss_batch_fun(self):
        """
        Return a function that computes the loss for a single parameter vector.
        Returns:
            Function that computes loss for a single theta vector.
        """
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.mean((y_pred - self.y_target) ** 2)
            return loss_value
        return loss_single
    
    def loss_batch(self, theta_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for a batch of parameter vectors.
        Args:
            theta_batch: Batch of parameter vectors.
        Returns:
            Loss values for each parameter vector in the batch.
        """

        loss_single = self.get_loss_batch_fun()
        
        batched_loss = torch.func.vmap(loss_single)
        self.loss_batch_cached = batched_loss(theta_batch)
        return self.loss_batch_cached

    
    def noisy_grad_balistic(self, theta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Compute the noisy gradient for the ballistic regime.
        """
        return self.grad(theta) + gamma 
    def noisy_grad_batcheq(self, theta: torch.Tensor, gamma: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute the noisy gradient for the batch equivalent regime.
        """
        return self.grad(theta) + gamma / tau**0.5
    def grad(self, theta_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the loss for a batch of parameter vectors.
        """

        loss_single = self.get_loss_batch_fun()
        grad_fn = torch.func.grad(loss_single)

        batched_grad = torch.func.vmap(grad_fn)
        self.grad_batch = batched_grad(theta_batch)

        if self.loss_batch_cached is None:
            self.loss_batch_cached = self.loss_batch(theta_batch)

        return self.grad_batch.detach()
    
    def hessian(self, theta_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the Hessian of the loss for a batch of parameter vectors.
        """
        if theta_batch is None:
            raise ValueError("Specifica theta_batch")
        assert theta_batch.ndim == 2, "theta_batch deve essere (B, total_params)"
        
        loss_single = self.get_loss_batch_fun()
        
        def grad_fn(theta_vec):
            return torch.func.grad(loss_single)(theta_vec)
        
        def hessian_single(theta_vec):
            hess = torch.func.jacrev(grad_fn)(theta_vec)
            return hess
        
        batched_hessian = torch.func.vmap(hessian_single)
        hessian_batch = batched_hessian(theta_batch)
        
        self.hessian_batch = hessian_batch
        return hessian_batch.detach()
    
    def Sigma_sqrt(self, theta):
        """
        Return the square root of the covariance matrix (identity for each sample).
        """
        return torch.eye(self.grad_batch.shape[1], device=theta.device).unsqueeze(0).expand(theta.shape[0], -1, -1).detach()
    
    def Diag_sigma(self, theta):
        """
        Return the diagonal of the covariance matrix (ones for each sample).
        """
        return torch.ones_like(self.grad_batch).detach()

    def square_root_var_z_squared(self, theta):
        """
        Return the square root of the variance of z squared (scaled identity for each sample).
        """
        return (2**0.5 * torch.eye(self.grad_batch.shape[1], device=theta.device)).unsqueeze(0).expand(theta.shape[0], -1, -1).detach()

    def term_b1_RMSProp_BatchEq(self, theta_batch, denom):
        """
        Compute the term for RMSProp SDE in the batch equivalent regime.
        """

        
        loss_single = self.get_loss_batch_fun()
        def grad_fn(theta_vec):
            return torch.func.grad(loss_single)(theta_vec)
        def diag_weighted_scalar(theta_vec, v_vec):
            n_params = theta_vec.shape[0]
            
            def get_diag_element(idx):
                all_indices = torch.arange(n_params, device=theta_vec.device)
                basis_vec = (all_indices == idx).to(theta_vec.dtype)
                _, col_i = torch.func.jvp(grad_fn, (theta_vec.clone(),), (basis_vec.clone(),))
                return torch.sum(col_i * basis_vec)
            indices = torch.arange(n_params, device=theta_vec.device)
            hess_diag = torch.func.vmap(get_diag_element)(indices)
            return torch.sum((v_vec ** 2) * hess_diag)

        grad_of_diag_fn = torch.func.grad(diag_weighted_scalar, argnums=0)

        batched_fn = torch.func.vmap(grad_of_diag_fn)

        return batched_fn(theta_batch, denom).detach()
    
    def _unpack_parameters(self, theta_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert theta_batch from (B, total_params) to a dict of parameter tensors with batch.
        """

        params_dict = {}
        offset = 0

        for name, shape in self.param_shapes:
            size = torch.tensor(shape).prod().item()
            param_flat = theta_batch[:, offset:offset + size]
            param_batch = param_flat.view(theta_batch.shape[0], *shape)
            params_dict[name] = param_batch
            offset += size

        return params_dict



class ShallowNN(base_nn):
    """
    Shallow neural network using torch.func for vectorized forward pass.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the shallow neural network and prepare data and layers.
        """

        X_train, y_train = load_and_preprocess_data(dataset = 'Housing')
        self.x_input = X_train.to(device)
        self.y_target = y_train.to(device)
        
        self.input_dim = X_train.shape[1]
        self.A = nn.Linear(self.input_dim, 3, bias=False).to(device)
        self.B = nn.Linear(3, 1).to(device)

        self.initial_weights = torch.cat([
            self.A.weight.flatten(),
            self.B.weight.flatten(),
            self.B.bias.flatten()
        ])
        self.network = nn.Sequential(self.A,
                              nn.ReLU(),
                              self.B).to(device)
        self.device = device

        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0

        for name, param in self.network.named_parameters():
            shape = param.shape
            size = param.numel()
            self.param_shapes.append((name, shape))
            self.param_sizes.append(size)
            self.total_params += size

        self.grad_batch = None
        self.loss_batch_cached = None
    
    def get_loss_batch_fun(self):
        """
        Return a function that computes the MSE loss for a single parameter vector.
        """
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.mean((y_pred - self.y_target) ** 2)
            return loss_value
        return loss_single



class MLP(base_nn):
    """
    Multi-layer perceptron using torch.func for vectorized forward pass.
    """


    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the MLP and prepare data and layers.
        """
        X_train, y_train = load_and_preprocess_data(dataset = 'BreastCancer')
        self.x_input = X_train.to(device)
        self.y_target = y_train.squeeze().long().to(device)
        
        self.input_dim = X_train.shape[1]
        hidden_dim = 20
        num_layers = 10  

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(self.input_dim, hidden_dim).to(device))
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim).to(device))
        self.linear_layers.append(nn.Linear(hidden_dim, 2).to(device))

        params_list = []
        for layer in self.linear_layers:
            params_list.append(layer.weight.flatten())
            params_list.append(layer.bias.flatten())
        self.initial_weights = torch.cat(params_list)

        modules_sequence = []
        for i, layer in enumerate(self.linear_layers):
            modules_sequence.append(layer)
            if i < len(self.linear_layers) - 1:
                modules_sequence.append(nn.ReLU())

        self.network = nn.Sequential(*modules_sequence).to(device)

        self.device = device

        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0

        for name, param in self.network.named_parameters():
            shape = param.shape
            size = param.numel()
            self.param_shapes.append((name, shape))
            self.param_sizes.append(size)
            self.total_params += size

        self.grad_batch = None
        self.loss_batch_cached = None

    def get_loss_batch_fun(self):
        """
        Return a function that computes the cross-entropy loss for a single parameter vector.
        """
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.nn.functional.cross_entropy(y_pred, self.y_target)
            return loss_value
        return loss_single
    
    def print_accuracy(self, theta_vec):
        """
        Print the accuracy of the model for a batch of parameter vectors.
        """
        def aux_function(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_input,))
            predicted_classes = torch.argmax(y_pred, dim=1)
            accuracy = (predicted_classes == self.y_target).float().mean()
            return accuracy

        batched_accuracy = torch.func.vmap(aux_function)
        accuracy_batched = batched_accuracy(theta_vec)
        print(f'Accuracy: {accuracy_batched.mean().item():.4f}')


class ResidualBlock(nn.Module):
    """
    Residual block: output = ReLU(conv(x) + x)
    """

    def __init__(self, conv_layer):
        """
        Initialize the residual block with a convolutional layer.
        """
        super().__init__()
        self.conv = conv_layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass for the residual block.
        """
        return self.relu(self.conv(x) + x)


class ResNet(base_nn):
    """
    Simple ResNet-like architecture for image classification tasks.
    """
    def __init__(self, dataset = 'MNIST', device: str = 'cuda' if torch.cuda.is_available() else 'cpu', Batch_for_hessian = 75):
        """
        Initialize the ResNet model and prepare data and layers.
        """
        if dataset == 'MNIST':
            in_channels = 1
            output_dim = 10
        elif dataset == 'CIFAR10':
            in_channels = 3
            output_dim = 10
        else:
            raise ValueError(f"Dataset {dataset} not supported for ResNet.")
        X_train, y_train = load_and_preprocess_data(dataset = dataset)
        self.x_input = X_train.to(device)
        self.y_target = y_train.squeeze().long().to(device)
    
        self.Batch_for_hessian = Batch_for_hessian
        print("Using Batch size for Hessian:", self.Batch_for_hessian)
        cutoff = (self.x_input.shape[0] // self.Batch_for_hessian) * self.Batch_for_hessian
        self.x_input = self.x_input[:cutoff]
        self.y_target = self.y_target[:cutoff]
        self.x_input_reshaped = self.x_input.reshape(self.Batch_for_hessian, -1, *self.x_input.shape[1:])
        self.y_target_reshaped = self.y_target.reshape(self.Batch_for_hessian, -1)

        hidden_channels = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=5, stride=3, padding=0).to(device)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1).to(device)
        self.res_block = ResidualBlock(self.conv2).to(device)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(hidden_channels, 16).to(device)
        self.fc2 = nn.Linear(16, output_dim).to(device)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.res_block,  
            self.pool,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2
        ).to(device)

        params_list = []
        for param in self.network.parameters():
            params_list.append(param.flatten()) 
        self.initial_weights = torch.cat(params_list)

        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0

        for name, param in self.network.named_parameters():
            shape = param.shape
            size = param.numel()
            self.param_shapes.append((name, shape))
            self.param_sizes.append(size)
            self.total_params += size

        self.grad_batch = None
        self.loss_batch_cached = None

    def get_loss_batch_fun(self):
        """
        Return a function that computes the cross-entropy loss for a single parameter vector.
        """
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.nn.functional.cross_entropy(y_pred, self.y_target)
            return loss_value
        return loss_single
    
    def hessian(self, theta_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the Hessian of the loss for a batch of parameter vectors using mini-batches.
        """
        if theta_batch is None:
            raise ValueError("Specifica theta_batch")
        assert theta_batch.ndim == 2, "theta_batch deve essere (B, total_params)"
        
        def loss_on_batch(theta_vec, x_batch, y_batch):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            
            y_pred = torch.func.functional_call(self.network, params_dict, (x_batch,))
            loss_value = torch.nn.functional.cross_entropy(y_pred, y_batch)
            return loss_value
        
        total_hessian = 0
        for i in range(self.Batch_for_hessian):
            x_b = self.x_input_reshaped[i]
            y_b = self.y_target_reshaped[i]

            def grad_fn(theta_vec):
                return torch.func.grad(loss_on_batch)(theta_vec, x_b, y_b)
            
            def hessian_single(theta_vec):
                hess = torch.func.jacrev(grad_fn)(theta_vec)
                return hess
            
            batched_hessian = torch.func.vmap(hessian_single)
            hessian_batch = batched_hessian(theta_batch)
            total_hessian += hessian_batch.detach()
        
            del hessian_batch
            del grad_fn
            del hessian_single
        
        self.hessian_batch = total_hessian / self.Batch_for_hessian
        return self.hessian_batch
    
    def term_b1_RMSProp_BatchEq(self, theta_batch, denom, chunk_size=24):
        """
        Compute the term for RMSProp SDE in the batch equivalent regime (chunked for memory efficiency).
        """
        torch.cuda.empty_cache()
        loss_single = self.get_loss_batch_fun()
        n_params = theta_batch.shape[1]
        device = theta_batch.device
        dtype = theta_batch.dtype
        
        all_indices = torch.arange(n_params, device=device)
        
        total_grad = torch.zeros_like(theta_batch)
        def grad_fn_single(theta_vec):
            return torch.func.grad(loss_single)(theta_vec)

        for i in range(0, n_params, chunk_size):
            end = min(i + chunk_size, n_params)
            
            def compute_chunk_score_per_sample(theta_vec, denom_vec):
                
                def get_diag_element(idx_in_chunk):
                    real_idx = i + idx_in_chunk
                    
                    basis_vec = (all_indices == real_idx).to(dtype)

                    _, col_i = torch.func.jvp(grad_fn_single, (theta_vec,), (basis_vec,))
                    
                    return torch.sum(col_i * basis_vec)
                
                chunk_indices = torch.arange(end - i, device=device)
                hess_diag_chunk = torch.func.vmap(get_diag_element)(chunk_indices)

                v_vec_chunk = denom_vec[i:end]
                
                return torch.sum((v_vec_chunk ** 2) * hess_diag_chunk)

            grad_single_fn = torch.func.grad(compute_chunk_score_per_sample, argnums=0)
            batched_grad_fn = torch.func.vmap(grad_single_fn, in_dims=(0, 0))
            partial_grad = batched_grad_fn(theta_batch, denom)
            
            total_grad += partial_grad.detach()
            del partial_grad
            
        return total_grad
