import torch
import torch.nn as nn
from .layers import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, bottleneck_channels, num_res_blocks):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.inv_conv = Invertible1x1Conv(in_channels)
        self.coupling = CNNCouplingLayer(
            in_channels, 
            hidden_channels, 
            bottleneck_channels=bottleneck_channels, 
            num_res_blocks=num_res_blocks
        )

    def forward(self, x):
        x, log_det_act = self.actnorm(x)
        x, log_det_conv = self.inv_conv(x)
        x, log_det_coup = self.coupling(x)
        return x, log_det_act + log_det_conv + log_det_coup

class DGLOWNetwork(nn.Module):
    def __init__(self, in_channels: int, num_levels: int, steps_per_level: int, hidden_channels: int, bottleneck_channels: int, num_res_blocks: int):
        super(DGLOWNetwork, self).__init__()
        self.squeeze = Squeeze()
        self.levels = nn.ModuleList()
        
        current_channels = in_channels
        for _ in range(num_levels):
            current_channels *= 4  # After Squeeze
            level_flows = nn.ModuleList([
                FlowStep(
                    current_channels, 
                    hidden_channels, 
                    bottleneck_channels=bottleneck_channels, 
                    num_res_blocks=num_res_blocks
                ) for _ in range(steps_per_level)
            ])
            self.levels.append(level_flows)
            
            split = Split(current_channels)
            self.levels.append(split)
            current_channels //= 2 # After Split
            
        self.levels = self.levels[:-1]  # Remove last split

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)

        total_log_det = torch.zeros(x.shape[0], device=x.device)
        
        # Initial Squeeze - not collected for deep supervision
        x, log_det = self.squeeze(x)
        total_log_det += log_det
        
        outputs = []
        
        for level in self.levels:
            if isinstance(level, nn.ModuleList): # Flow steps
                for flow_step in level:
                    x, log_det = flow_step(x)
                    total_log_det += log_det
                    outputs.append((x.flatten(start_dim=1), total_log_det.clone()))

            elif isinstance(level, Split): # Split
                x, z = level(x)

        return outputs
