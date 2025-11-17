import torch
import torch.nn as nn
from .layers import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, bottleneck_channels, num_res_blocks, actnorm_initialization="identity", invconv_initialization="orthogonal"):
        super().__init__()
        self.actnorm = ActNorm(in_channels, initialization=actnorm_initialization)
        self.inv_conv = Invertible1x1Conv(in_channels, initialization=invconv_initialization)
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
    def __init__(self, in_channels: int, num_levels: int, steps_per_level: int, hidden_channels: int, bottleneck_channels: int, num_res_blocks: int, actnorm_initialization: str = "identity", invconv_initialization: str = "orthogonal"):
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
                    num_res_blocks=num_res_blocks,
                    actnorm_initialization=actnorm_initialization,
                    invconv_initialization=invconv_initialization
                ) for _ in range(steps_per_level)
            ])
            self.levels.append(level_flows)
            
            split = Split(current_channels)
            self.levels.append(split)
            current_channels //= 2  # After Split

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
                    
                    x_out, log_det = self.squeeze.inverse(x)
                    for z in z_out:
                        x_out = torch.cat([x_out, z], dim=1)
                        x_out, log_det = self.squeeze.inverse(x_out)
                    outputs.append((x_out, total_log_det.clone() + log_det))
            elif isinstance(level, Split): # Split
                x, z = level(x)
                z_out.append(z)

        return outputs
    
    def inverse(self, z_final):
        total_log_det = torch.zeros(z_final.shape[0], device=z_final.device)
        zs = []
        x = z_final
        
        for level in self.levels:
            if isinstance(level, Split): # Split
                x, z = level(x)
                zs.append(z)
            elif isinstance(level, nn.ModuleList): # Flow steps
                x, _ = self.squeeze(x)
        
        for level in reversed(self.levels):
            if isinstance(level, nn.ModuleList): # Flow steps
                for flow_step in reversed(level):
                    x, log_det = flow_step.inverse(x)
                    total_log_det += log_det
                    
                x, log_det_squeeze = self.squeeze.inverse(x)
                total_log_det += log_det_squeeze
            elif isinstance(level, Split): # Split
                z = zs.pop()
                x, log_det = level.inverse(x, z)
                total_log_det += log_det
                
        return x, total_log_det
    
    @property
    def total_supervision_layers(self):
        return self.num_levels * self.steps_per_level


if __name__ == "__main__":
    # --- Test FlowStep Invertibility ---
    print("--- Testing FlowStep Invertibility ---")
    x_step = torch.randn(8, 2, 14, 14)
    step = FlowStep(
        in_channels=2,
        hidden_channels=64,
        bottleneck_channels=16,
        num_res_blocks=3,
        actnorm_initialization="data-dependent",
        invconv_initialization="orthogonal"
    )
    z, log_det_fwd = step(x_step)
    x_recon_step, log_det_inv = step.inverse(z)
    
    recon_error_step = torch.abs(x_step - x_recon_step).mean().item()
    log_det_error_step = torch.abs(log_det_fwd + log_det_inv).mean().item()

    print(f"FlowStep Reconstruction Error: {recon_error_step:.2e}")
    print(f"FlowStep Log-Determinant Sum: {log_det_error_step:.2e}")
    if recon_error_step < 1e-5 and log_det_error_step < 1e-5:
        print("[STATUS] PASSED")
    else:
        print("[STATUS] FAILED")


    # --- Test DGLOWNetwork Invertibility ---
    print("\n--- Testing DGLOWNetwork Invertibility ---")
    model = DGLOWNetwork(
        in_channels=1,
        num_levels=2,
        steps_per_level=1,
        hidden_channels=64,
        bottleneck_channels=16,
        num_res_blocks=3,
        actnorm_initialization="data-dependent",
        invconv_initialization="orthogonal"
    )
    x_net = torch.randn(8, 1, 28, 28)
    outputs = model(x_net)
    
    # The final output of the forward pass is the reconstructed image
    x_recon_fwd = outputs[-1][0]
    
    # The inverse method should also reconstruct the image
    x_recon_inv, _ = model.inverse(x_recon_fwd)
    
    recon_error_net = torch.abs(x_net - x_recon_inv).mean().item()
    print(f"DGLOWNetwork Reconstruction Error: {recon_error_net:.2e}")
    if recon_error_net < 1e-5:
        print("[STATUS] PASSED")
    else:
        print("[STATUS] FAILED")