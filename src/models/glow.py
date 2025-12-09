import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from src.models.modules import AffineCouplingLayer, Squeeze, Split, LogitTransform


class Glow(nn.Module):
    def __init__(self, in_channels: int, input_shape: tuple[int, int], num_levels: int, steps_per_level: list[int], hidden_channels: int, scale_clamp: float, checkpoint_grads: bool = False):
        super(Glow, self).__init__()
        assert len(steps_per_level) == num_levels, "steps_per_level length must match num_levels"
        
        self.squeeze = Squeeze()
        self.logit_transform = LogitTransform(alpha=0.05)
        self.split_levels = nn.ModuleList()
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level  
        self.checkpoint_grads = checkpoint_grads
        self.output_shapes = []
        
        C = in_channels
        H, W = input_shape
        for level_idx in range(num_levels):
            C *= 4  # After Squeeze
            H //= 2
            W //= 2
            level_flows = nn.ModuleList([
                AffineCouplingLayer(
                    C, 
                    hidden_channels, 
                    scale_clamp=scale_clamp
                ) for _ in range(self.steps_per_level[level_idx])
            ])
            self.split_levels.append(level_flows)
            
            if level_idx < num_levels - 1:
                split = Split()
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
                z, h = level(z, method="split")
                outs.append((z, h))

        outs.append((None, z))  # Final latent without split
        log_dets = torch.stack(log_dets, dim=0)
        return outs, log_dets
    
    def inverse(self, z):
        # LEGACY: used to reconstruct from a list of tensors [z0, z1, ..., zN]
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
    # print("--- Testing FlowStep Invertibility ---")
    # x = torch.randn(8, 3, 32, 32).clamp(min=0.0, max=1.0)
    # step = AffineCouplingLayer(
    #     in_channels=3,
    #     hidden_channels=64,
    #     scale_clamp=1.0,
    # )
    # z, log_det = step(x)
    # x_recon, log_det_inv = step.inverse(z)
    
    # recon_error_step = torch.abs(x - x_recon).mean().item()
    # log_det_error_step = torch.abs(log_det + log_det_inv).mean().item()

    # print(f"FlowStep Reconstruction Error: {recon_error_step:.2e}")
    # print(f"FlowStep Log-Determinant Sum: {log_det_error_step:.2e}")
    # if recon_error_step < 1e-5 and log_det_error_step < 1e-5:
    #     print("[STATUS] PASSED")
    # else:
    #     print("[STATUS] FAILED")


    # --- Test DGLOWNetwork Invertibility ---
    print("\n--- Testing DGLOWNetwork Invertibility ---")
    model = Glow(
        in_channels=3,
        input_shape=(32, 32),
        num_levels=3,
        steps_per_level=[4, 4, 4],
        hidden_channels=64,
        scale_clamp=1.0,
    )
    x = torch.randn(8, 3, 32, 32).clamp(min=0.0, max=1.0)
    z, log_det = model(x)
    print(f"Output Latent Shapes: {[zi[1].shape for zi in z]}")
    print(model.output_shapes)
    
    # # The inverse method should also reconstruct the image
    # x_recon, log_det_inv = model.inverse(z)

    # recon_error_net = torch.abs(x - x_recon).mean().item()
    # print(f"DGLOWNetwork Reconstruction Error: {recon_error_net:.2e}")
    # if recon_error_net < 1e-5:
    #     print("[STATUS] PASSED")
    # else:
    #     print("[STATUS] FAILED")