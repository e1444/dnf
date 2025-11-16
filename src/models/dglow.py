import torch
import torch.nn as nn
from layers import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split

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
    
    def inverse(self, z):
        z, log_det_coup = self.coupling.inverse(z)
        z, log_det_conv = self.inv_conv.inverse(z)
        z, log_det_act = self.actnorm.inverse(z)
        return z, log_det_act + log_det_conv + log_det_coup

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
    
    def inverse(self, z_final):
        """
        Generates data by applying the inverse transformation from the full latent vector z_final.
        """
        total_log_det = torch.zeros(z_final.shape[0], device=z_final.device)
        
        # We need to know the shapes of the tensors that were split off.
        # We can infer them from the architecture.
        z_shapes = []
        c, h, w = 1, 28, 28 # Initial MNIST shape
        for i in range(self.num_levels):
            c, h, w = c * 4, h // 2, w // 2 # Squeeze
            if i < self.num_levels - 1:
                z_shapes.append((c // 2, h, w)) # Shape of the split-off z
                c = c // 2 # Update channel count for the next level
            else:
                z_shapes.append((c, h, w)) # Final x shape
        
        # Split the flat z_final vector back into its constituent parts
        split_sizes = [s[0]*s[1]*s[2] for s in z_shapes]
        z_parts = z_final.split(split_sizes, dim=1)
        
        # The last part of z_final corresponds to the active tensor 'x' before the final split.
        x = z_parts[-1].view(z_final.shape[0], *z_shapes[-1])
        
        # Stack of z's that were split off, to be used in reverse.
        z_stack = [p.view(z_final.shape[0], *s) for p, s in zip(z_parts[:-1], z_shapes[:-1])]

        # Iterate through levels in reverse
        for i in reversed(range(self.num_levels)):
            # Get the correct module list for flows and the split layer
            flow_level = self.levels[i*2]
            
            # 1. Inverse Flow Steps
            for flow_step in reversed(flow_level):
                x, log_det = flow_step.inverse(x)
                total_log_det += log_det
            
            # 2. Inverse Squeeze
            x, log_det = self.squeeze.inverse(x)
            total_log_det += log_det
            
            # 3. Inverse Split (if not the first level)
            if i > 0:
                split_level = self.levels[i*2 - 1]
                z_part = z_stack.pop()
                x, log_det = split_level.inverse(x, z_part)
                total_log_det += log_det
                
        return x, total_log_det
    
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
        
    print(outputs[-1][0][0].shape)
    
    x = model.inverse(outputs[-1][0])