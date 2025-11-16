import torch
import torch.nn as nn
from .layers import ActNorm, Squeeze, Invertible1x1Conv, CNNCouplingLayer

class DNFNetwork(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int, bottleneck_channels: int, num_res_blocks: int):
        super(DNFNetwork, self).__init__()
        self.squeeze = Squeeze()
        current_channels = in_channels * 4
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        
        for _ in range(num_layers):
            self.layers.append(ActNorm(current_channels))
            self.layers.append(Invertible1x1Conv(current_channels))
            self.layers.append(CNNCouplingLayer(current_channels, hidden_channels=hidden_channels, bottleneck_channels=bottleneck_channels, num_res_blocks=num_res_blocks))

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)

        total_log_det = torch.zeros(x.shape[0], device=x.device)
        x, log_det = self.squeeze(x)
        total_log_det += log_det
        intermediate_outputs = []

        for layer in self.layers:
            x, log_det = layer(x)
            total_log_det += log_det
            if isinstance(layer, CNNCouplingLayer):
                intermediate_outputs.append((x.flatten(start_dim=1), total_log_det.clone()))

        return intermediate_outputs
    
    def inverse(self, z):
        """
        Generates data by applying the inverse transformation from the latent space z back to the data space.
        
        Args:
            z (torch.Tensor): A tensor from the latent space, typically of shape 
                              (batch_size, channels, height, width).
        
        Returns:
            A tuple (x, total_log_det):
                x (torch.Tensor): The generated data tensor.
                total_log_det (torch.Tensor): The log-determinant of the inverse transformation.
        """
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        # Apply layers in reverse order
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            total_log_det += log_det

        # Apply inverse squeeze
        x, log_det = self.squeeze.inverse(z)
        total_log_det += log_det

        return x, total_log_det
    
    @property
    def total_supervision_layers(self):
        return self.num_layers
