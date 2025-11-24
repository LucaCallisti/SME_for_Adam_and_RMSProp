import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ShallowNN:
    """
    Usa torch.func per vettorializzare automaticamente il forward pass.
    Funziona con reti complesse senza modifiche.
    """

    def __init__(self, input_dim, mid_dim, output_dim, dataset, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.A = nn.Linear(input_dim, mid_dim, bias=False).to(device)
        self.B = nn.Linear(mid_dim, output_dim).to(device)

        self.initial_weights = torch.cat([
            self.A.weight.flatten(),
            self.B.weight.flatten(),
            self.B.bias.flatten()
        ])
        self.network = nn.Sequential(self.A,
                              nn.ReLU(),
                              self.B).to(device)
        self.device = device

        # Calcola total_params e estrai param shapes
        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0

        for name, param in self.network.named_parameters():
            shape = param.shape
            size = param.numel()
            self.param_shapes.append((name, shape))
            self.param_sizes.append(size)
            self.total_params += size

        self.x_input = dataset.X.to(device)
        self.y_target = dataset.y.to(device)

        self.x_val = dataset.X_val.to(device)
        self.y_val = dataset.y_val.to(device)

        self.grad_batch = None
        self.loss_batch_cached = None

    def forward_batch(self, x: torch.Tensor, theta_batch: torch.Tensor) -> torch.Tensor:
        """
        Applica la rete con batch di parametri usando torch.func.

        Args:
            x: input fisso
            theta_batch: (B, total_params)

        Returns:
            output di shape (B, output_dim)
        """
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+. Usa BatchParameterNetwork invece.")

        def forward_single(theta_vec):
            # Converte theta_vec in dict di parametri
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            # Estrai il primo (e unico) elemento del batch
            params_dict = {k: v[0] for k, v in params_dict.items()}
            return F.functional_call(self.network, params_dict, (x,))

        # Vmap sul primo asse di theta_batch
        batched_forward = F.vmap(forward_single)
        return batched_forward(theta_batch)
    
    def val_loss_batch(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+.")
        
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = F.functional_call(self.network, params_dict, (self.x_val,))
            loss_value = torch.mean((y_pred - self.y_val) ** 2)
            return loss_value
        
        # Vmap sul primo asse di theta_batch
        batched_loss = F.vmap(loss_single)
        val_loss = batched_loss(theta_batch)
        return val_loss
    
    def loss_batch(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+.")

        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = F.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.mean((y_pred - self.y_target) ** 2)
            return loss_value
        
        # Vmap sul primo asse di theta_batch
        batched_loss = F.vmap(loss_single)
        self.loss_batch_cached = batched_loss(theta_batch)
        return self.loss_batch_cached
    
    def noisy_grad(self, theta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self.grad(theta) + gamma 
    def grad(self, theta_batch: torch.Tensor) -> torch.Tensor:
        try:
            import torch.func as F
        except ImportError:
            raise RuntimeError("torch.func richiede PyTorch 2.0+. Usa BatchParameterNetwork invece.")

        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = F.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.mean((y_pred - self.y_target) ** 2)
            return loss_value

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
        
        def loss_single(theta_vec):
            params_dict = self._unpack_parameters(theta_vec.unsqueeze(0))
            params_dict = {k: v[0] for k, v in params_dict.items()}
            y_pred = F.functional_call(self.network, params_dict, (self.x_input,))
            loss_value = torch.mean((y_pred - self.y_target) ** 2)
            return loss_value
        
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

    def term_batch_eq_regime(self):
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


# Esempio di utilizzo
if __name__ == "__main__":
    torch.manual_seed(42)
    input_dim = 3
    output_dim = 1
    N = 5  # numero di esempi
    X = torch.randn(N, input_dim)  # shape (N, input_dim)
    Y = torch.randn(N, output_dim) # shape (N, output_dim)

    mid_dim = 3
    # Crea un dataset wrapper semplice
    class SimpleDataset:
        def __init__(self, X, y):
            self.X = X
            self.y = y
    
    dataset = SimpleDataset(X, Y)

    model = SimpleNN(input_dim, mid_dim, output_dim, dataset, device='cpu')
    params = model.get_weights()
    Y_pred = model(X)
    loss = model.loss(X, Y)
    print("Loss MSE:", loss.item())

    # Gradienti autograd concatenati (metodo classico)
    grad_vector = model.autograd_vector()
    print("\nVettore gradiente autograd (classico):")
    print(grad_vector)

    # Hessiano
    hess = model.hessian_matrix()
    print("\nHessiano:")
    print(hess)

    print("\n" + "=" * 60)
    print("VMAPPED PARAMETER NETWORK EXAMPLE")
    print("=" * 60)
    
    
    vmapped_model = VmappedParameterNetwork(input_dim, mid_dim, output_dim, dataset, device=args.device)    
    theta_batch = params.unsqueeze(0).expand(4, -1).clone() 
    
    
    # Esempio 2: Calcolo gradienti per un batch di parametri
    print("-" * 60)
    print("ESEMPIO 2: Gradienti per batch di parametri")
    print("-" * 60)
    grad_batch = vmapped_model.autograd_vector_batch(theta_batch)
    print(f"Gradient batch shape: {grad_batch.shape}")
    print(f"Gradient batch:\n{grad_batch}\n")

    # Step 3: Calcola Hessiano ESATTO completo
    print("-" * 60)
    print("STEP 3: Hessiano ESATTO completo batch-wise")
    print("-" * 60)
    hessian_batch = vmapped_model.hessian_matrix_batch(theta_batch)
    print(f"Hessian batch shape: {hessian_batch.shape}")
    print(f"Hessian:\n{hessian_batch}")
    

