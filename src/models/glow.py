import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .modules import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split, LogitTransform

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dropout: float,
        actnorm_init="identity",
        invconv_init="orthogonal",
        *,
        actnorm_init_std: torch.Tensor
    ):
        super().__init__()
        self.actnorm = ActNorm(
            in_channels,
            initialization=actnorm_init,
            init_std=actnorm_init_std
        )
        self.inv_conv = Invertible1x1Conv(
            in_channels,
            initialization=invconv_init
        )
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
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_levels: int,
        steps_per_level: list[int],
        std_per_level: list[torch.Tensor],
        hidden_channels: int,
        num_blocks: int,
        dropout: float,
        actnorm_initialization: str = "data-dependent",
        invconv_initialization: str = "orthogonal",
        checkpoint_grads: bool = False
    ):
        super(DGLOWNetwork, self).__init__()
        assert len(steps_per_level) == num_levels, "steps_per_level length must match num_levels"
        
        self.squeeze = Squeeze()
        self.logit_transform = LogitTransform(alpha=0.05)
        self.split_levels = nn.ModuleList()
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level  
        self.checkpoint_grads = checkpoint_grads
        
        C, H, W = input_shape
        for level_idx in range(num_levels):
            C *= 4  # After Squeeze
            H //= 2
            W //= 2
            
            std_per_step = [
                torch.ones(C, device=std_per_level[0].device) * torch.mean(std_per_level[level_idx])
                for _ in range(self.steps_per_level[level_idx])
            ]
            if level_idx == num_levels - 1:
                std_per_step[-1] = std_per_level[level_idx]
            else:
                std_per_step[-1][C // 2:] = std_per_level[level_idx]
            
            level_flows = nn.ModuleList([
                FlowStep(
                    C, 
                    hidden_channels, 
                    num_blocks=num_blocks,
                    dropout=dropout,
                    actnorm_init=actnorm_initialization,
                    actnorm_init_std=std_per_step[i],
                    invconv_init=invconv_initialization
                ) for i in range(self.steps_per_level[level_idx])
            ])
            self.split_levels.append(level_flows)
            
            if level_idx < num_levels - 1:
                split = Split(C)
                self.split_levels.append(split)
                C //= 2  # After Split
    
    def forward(self, x):
        log_dets = []
        z, log_det = self.logit_transform(x)
        log_dets.append(log_det)
        
        outs = []
        level_log_det = torch.zeros(x.size(0), device=x.device)
        for level in self.split_levels:
            if isinstance(level, nn.ModuleList): # Flow steps
                z, log_det = self.squeeze(z)
                level_log_det = level_log_det + log_det
                
                for flow_step in level:
                    if self.checkpoint_grads and z.requires_grad:
                        z, log_det = checkpoint.checkpoint(flow_step, z, use_reentrant=False)   # type: ignore
                    else:
                        z, log_det = flow_step(z)
                    level_log_det = level_log_det + log_det
            elif isinstance(level, Split): # Split
                z, h = level(z)
                outs.append((z, h))
                log_dets.append(level_log_det)
                level_log_det = torch.zeros(x.size(0), device=x.device)

        outs.append((None, z))  # Final latent without split
        log_dets.append(level_log_det)
        log_dets = torch.stack(log_dets, dim=0)
        return outs, log_dets
    
    def inverse(self, z):
        total_log_det = torch.zeros(z[0].shape[0], device=z[0].device)
        x = z.pop()
        
        for level in reversed(self.split_levels):
            if isinstance(level, nn.ModuleList): # Flow steps
                for flow_step in reversed(level):
                    x, log_det = flow_step.inverse(x)   # type: ignore
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
    
    @staticmethod
    def output_shapes(input_shape: tuple[int, int, int], num_levels: int):
        C, H, W = input_shape
        output_shapes = []
        for level_idx in range(num_levels):
            C *= 4  # After Squeeze
            H //= 2
            W //= 2
            if level_idx < num_levels - 1:
                C //= 2  # After Split
            output_shapes.append((C, H, W))
        return output_shapes


if __name__ == "__main__":
    pass