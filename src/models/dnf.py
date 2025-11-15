import torch
import torch.nn as nn
from .layers import ActNorm, Squeeze, Invertible1x1Conv, CNNCouplingLayer

class DNFNetwork(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int, bottleneck_channels: int, num_res_blocks: int):
        super(DNFNetwork, self).__init__()
        self.squeeze = Squeeze()
        current_channels = in_channels * 4
        self.layers = nn.ModuleList()
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
