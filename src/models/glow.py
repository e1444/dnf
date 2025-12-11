import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .layers import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split, LogitTransform

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_blocks: int, dropout: float, actnorm_initialization="identity", invconv_initialization="orthogonal"):
        super().__init__()
        self.actnorm = ActNorm(in_channels, initialization=actnorm_initialization)
        self.inv_conv = Invertible1x1Conv(in_channels, initialization=invconv_initialization)
        self.coupling = CNNCouplingLayer(
            in_channels, 
            hidden_channels,
            num_blocks=num_blocks,
            dropout=dropout
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
    def __init__(self, input_shape: tuple[int, int, int], num_levels: int, steps_per_level: list[int], hidden_channels: int, num_blocks: int, dropout: float, actnorm_initialization: str = "data-dependent", invconv_initialization: str = "orthogonal", checkpoint_grads: bool = False):
        super(DGLOWNetwork, self).__init__()
        assert len(steps_per_level) == num_levels, "steps_per_level length must match num_levels"
        
        self.squeeze = Squeeze()
        self.logit_transform = LogitTransform(alpha=0.05)
        self.split_levels = nn.ModuleList()
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level  
        self.checkpoint_grads = checkpoint_grads
        self.output_shapes = []
        
        C, H, W = input_shape
        for level_idx in range(num_levels):
            C *= 4  # After Squeeze
            H //= 2
            W //= 2
            level_flows = nn.ModuleList([
                FlowStep(
                    C, 
                    hidden_channels, 
                    num_blocks=num_blocks,
                    dropout=dropout,
                    actnorm_initialization=actnorm_initialization,
                    invconv_initialization=invconv_initialization
                ) for _ in range(self.steps_per_level[level_idx])
            ])
            self.split_levels.append(level_flows)
            
            if level_idx < num_levels - 1:
                split = Split(C)
                self.split_levels.append(split)
                C //= 2  # After Split
                
            self.output_shapes.append((C, H, W))
    
    def forward(self, x):
        log_dets = []
        z, log_det = self.logit_transform(x)
        log_dets.append(log_det)
        
        outs = []
        for level in self.split_levels:
            if isinstance(level, nn.ModuleList): # Flow steps
                z, log_det = self.squeeze(z)
                log_dets.append(log_det)
                
                for flow_step in level:
                    if self.checkpoint_grads and z.requires_grad:
                        z, log_det = checkpoint.checkpoint(flow_step, z, use_reentrant=False)
                    else:
                        z, log_det = flow_step(z)
                    log_dets.append(log_det)
            elif isinstance(level, Split): # Split
                z, h = level(z)
                outs.append((z, h))

        outs.append((None, z))  # Final latent without split
        log_dets = torch.stack(log_dets, dim=0)
        return outs, log_dets
    
    def inverse(self, z):
        total_log_det = torch.zeros(z[0].shape[0], device=z[0].device)
        x = z.pop()
        
        for level in reversed(self.split_levels):
            if isinstance(level, nn.ModuleList): # Flow steps
                for flow_step in reversed(level):
                    x, log_det = flow_step.inverse(x)
                    total_log_det += log_det
                    
                x, log_det_squeeze = self.squeeze.inverse(x)
                total_log_det += log_det_squeeze
            elif isinstance(level, Split): # Split
                x, log_det = level.inverse(x, z.pop())
                total_log_det += log_det
                
        x, log_det = self.logit_transform.inverse(x)
        total_log_det += log_det

        return x, total_log_det
    
    @property
    def total_steps(self):
        return sum(self.steps_per_level)


if __name__ == "__main__":
    # --- Test FlowStep Invertibility ---
    print("--- Testing FlowStep Invertibility ---")
    x_step = torch.randn(8, 2, 14, 14)
    step = FlowStep(
        in_channels=2,
        hidden_channels=64,
        num_blocks=1,
        dropout=0.0,
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
        num_levels=3,
        steps_per_level=[4, 4, 4],
        hidden_channels=64,
        num_blocks=1,
        dropout=0.0,
        actnorm_initialization="identity",
        invconv_initialization="identity"
    )
    x = torch.randn(8, 1, 32, 32).clamp(min=0.0, max=1.0)
    outputs = model(x)
    
    # The final output of the forward pass is the reconstructed image
    z_parts, log_det = outputs[-1]
    
    # The inverse method should also reconstruct the image
    x_recon, log_det_inv = model.inverse(z_parts)

    recon_error_net = torch.abs(x - x_recon).mean().item()
    print(f"DGLOWNetwork Reconstruction Error: {recon_error_net:.2e}")
    if recon_error_net < 1e-5:
        print("[STATUS] PASSED")
    else:
        print("[STATUS] FAILED")