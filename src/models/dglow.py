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
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level
        
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
        outputs = []
        z_out = []
        
        for level in self.levels:
            if isinstance(level, nn.ModuleList): # Flow steps
                x, log_det_squeeze = self.squeeze(x)
                total_log_det += log_det_squeeze
                
                for flow_step in level:
                    x, log_det = flow_step(x)
                    total_log_det += log_det
                    
                    x_out = x.flatten(start_dim=1)
                    x_out = torch.cat([x_out] + z_out, dim=1)
                    outputs.append((x_out, total_log_det.clone()))

            elif isinstance(level, Split): # Split
                x, z = level(x)
                z_out.append(z.flatten(start_dim=1))

        return outputs
    
    @property
    def total_supervision_layers(self):
        return self.num_levels * self.steps_per_level

if __name__ == "__main__":
    model = DGLOWNetwork(
        in_channels=1,
        num_levels=2,
        steps_per_level=6,
        hidden_channels=156,
        bottleneck_channels=128,
        num_res_blocks=3
    )
    x = torch.randn(16, 1, 28, 28)
    outputs = model(x)
    for i, (z, log_det) in enumerate(outputs):
        print(f"Output {i}: z shape = {z.shape}, log_det shape = {log_det.shape}")