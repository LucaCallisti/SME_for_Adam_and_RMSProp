import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from NeuralNetwork.Utils import load_and_preprocess_data


class base_nn:
    def forward_batch(self, x: torch.Tensor, theta_batch: torch.Tensor) -> torch.Tensor:
        """
        Applica la rete con batch di parametri usando torch.func.

        Args:
            x: input fisso
            theta_batch: (B, total_params)

        Returns:
            output di shape (B, output_dim)
        """

        def forward_single(theta_vec):
            # Converte theta_vec in dict di parametri
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            # Estrai il primo (e unico) elemento del batch
            params_dict = {k: v[0] for k, v in params_dict.items()}
            return torch.func.functional_call(self.network, params_dict, (x,))

        # Vmap sul primo asse di theta_batch
        batched_forward = F.vmap(forward_single)
        return batched_forward(theta_batch)

    def get_loss_batch_fun(self):
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_val,))
            loss_value = torch.mean((y_pred - self.y_val) ** 2)
            return loss_value
        return loss_single
    
    def loss_batch(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+.")

        loss_single = self.get_loss_batch_fun()
        
        # Vmap sul primo asse di theta_batch
        batched_loss = F.vmap(loss_single)
        self.loss_batch_cached = batched_loss(theta_batch)
        return self.loss_batch_cached
    
    def val_loss_batch(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+.")
        
        loss_single = self.get_loss_batch_fun()
        
        # Vmap sul primo asse di theta_batch
        batched_loss = F.vmap(loss_single)
        val_loss = batched_loss(theta_batch)
        return val_loss
    
    def noisy_grad_balistic(self, theta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self.grad(theta) + gamma 
    def noisy_grad_batcheq(self, theta: torch.Tensor, gamma: torch.Tensor, tau: float) -> torch.Tensor:
        return self.grad(theta) + gamma / tau**0.5
    def grad(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+. Usa BatchParameterNetwork invece.")

        loss_single = self.get_loss_batch_fun()
        grad_fn = F.grad(loss_single)

        # Vmap sul primo asse di theta_batch
        batched_grad = F.vmap(grad_fn)
        self.grad_batch = batched_grad(theta_batch)

        # Calcola anche la loss se non è stata calcolata
        if self.loss_batch_cached is None:
            self.loss_batch_cached = self.loss_batch(theta_batch)

        return self.grad_batch
    
    def hessian(self, theta_batch: torch.Tensor = None) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+.")
        
        if theta_batch is None:
            raise ValueError("Specifica theta_batch")
        assert theta_batch.ndim == 2, "theta_batch deve essere (B, total_params)"
        
        loss_single = self.get_loss_batch_fun()
        
        # Definisci la funzione gradiente
        def grad_fn(theta_vec):
            return F.grad(loss_single)(theta_vec)
        
        # Calcola l'Hessiano come Jacobiano del gradiente
        def hessian_single(theta_vec):
            # jacobian restituisce la matrice Jacobiana (total_params, total_params)
            hess = F.jacrev(grad_fn)(theta_vec)
            return hess
        
        # Vmap per calcolare l'Hessiano per ogni campione del batch
        batched_hessian = F.vmap(hessian_single)
        hessian_batch = batched_hessian(theta_batch)
        
        self.hessian_batch = hessian_batch
        return hessian_batch
    
    def Sigma_sqrt(self, theta):
        return torch.eye(self.grad_batch.shape[1], device=theta.device).unsqueeze(0).expand(theta.shape[0], -1, -1)
    
    def Diag_sigma(self, theta):
        return torch.ones_like(self.grad_batch)

    def square_root_var_z_squared(self, theta):
        return (2**0.5 * torch.eye(self.grad_batch.shape[1], device=theta.device)).unsqueeze(0).expand(theta.shape[0], -1, -1)

    def term_b1_RMSProp_BatchEq(self, theta):
        result = self.hessian_batch * torch.diag_embed(self.grad_batch) / (self.loss_batch_cached**0.5).view(-1, 1, 1) - torch.einsum('bi,bj->bij', self.grad_batch, self.grad_batch**2) / (4 * self.loss_batch_cached**1.5).view(-1, 1, 1)
        result[:, :, -1] = torch.zeros_like(result[:, :, -1])  
        return result

    def _unpack_parameters(self, theta_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Converte theta_batch da (B, total_params) a dict di parametri con batch.
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
    Usa torch.func per vettorializzare automaticamente il forward pass.
    Funziona con reti complesse senza modifiche.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # Creating datasets
        X_train, X_val, y_train, y_val = load_and_preprocess_data(dataset = 'Housing', test_size = 0.2)
        self.x_input = X_train.to(device)
        self.y_target = y_train.to(device)

        self.x_val = X_val.to(device)
        self.y_val = y_val.to(device)
        
        # Inizializing layers
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

        # Calculating parameter shapes and sizes for unpacking
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
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_val,))
            loss_value = torch.mean((y_pred - self.y_val) ** 2)
            return loss_value
        return loss_single



class MLP(base_nn):
    """
    Usa torch.func per vettorializzare automaticamente il forward pass.
    Funziona con reti complesse senza modifiche.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
         # Creating datasets
        X_train, X_val, y_train, y_val = load_and_preprocess_data(dataset = 'BreastCancer', test_size = 0.2)
        self.x_input = X_train.to(device)
        self.y_target = y_train.squeeze().long().to(device)

        self.x_val = X_val.to(device)
        self.y_val = y_val.squeeze().long().to(device)
        
        # Inizializing layers
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

        # Calculating parameter shapes and sizes for unpacking
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
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_val,))
            loss_value = torch.nn.functional.cross_entropy(y_pred, self.y_val)
            return loss_value
        return loss_single


class ResidualBlock(nn.Module):
    """Blocco residuale: output = ReLU(conv(x) + x)"""
    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ResNet(base_nn):
    def __init__(self, dataset = 'MNIST', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # Creating datasets
        if dataset == 'MNIST':
            input_height, input_width = 28, 28
            in_channels = 1
            output_dim = 10
        elif dataset == 'CIFAR10':
            input_height, input_width = 32, 32
            in_channels = 3
            output_dim = 10
        else:
            raise ValueError(f"Dataset {dataset} not supported for ResNet.")
        X_train, X_val, y_train, y_val = load_and_preprocess_data(dataset = dataset, test_size = 0.2)
        self.x_input = X_train.to(device)
        self.y_target = y_train.squeeze().long().to(device)

        self.x_val = X_val.to(device)
        self.y_val = y_val.squeeze().long().to(device)
        
        # Inizializing layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1).to(device)
        self.res_block = ResidualBlock(self.conv2).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        final_h = input_height // 2
        final_w = input_width // 2
        self.flat_dim = 32 * final_h * final_w

        self.fc1 = nn.Linear(self.flat_dim, 128).to(device)
        self.fc2 = nn.Linear(128, output_dim).to(device)

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.res_block,  # Residual: ReLU(conv2(x) + x)
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

        # Calculating parameter shapes and sizes for unpacking
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
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = torch.func.functional_call(self.network, params_dict, (self.x_val,))
            loss_value = torch.nn.functional.cross_entropy(y_pred, self.y_val)
            return loss_value
        return loss_single
    




