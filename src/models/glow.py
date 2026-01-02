import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from src.models.modules import ActNorm, Invertible1x1Conv, AffineCoupling, BlockAutoregressiveCoupling, Squeeze, Split, LogitTransform

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_resnet_blocks: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.actnorm = ActNorm(
            in_channels,
        )
        
        self.inv_conv = Invertible1x1Conv(
            in_channels,
        )
        
        self.coupling1 = AffineCoupling(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_resnet_blocks=num_resnet_blocks,
            dropout=dropout,
        )
        
        self.coupling2 = BlockAutoregressiveCoupling(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_resnet_blocks=num_resnet_blocks,
            block_size=block_size,
            dropout=dropout
        )

    def forward(self, x):
        x, log_det_act = self.actnorm(x)
        x, log_det_conv = self.inv_conv(x)
        x, log_det_coup1 = self.coupling1(x)
        x, log_det_coup2 = self.coupling2(x)
        return x, log_det_act + log_det_conv + log_det_coup1 + log_det_coup2
    
    def inverse(self, z):
        z, log_det_coup2 = self.coupling2.inverse(z)
        z, log_det_coup1 = self.coupling1.inverse(z)
        z, log_det_conv = self.inv_conv.inverse(z)
        z, log_det_act = self.actnorm.inverse(z)
        return z, log_det_act + log_det_conv + log_det_coup1 + log_det_coup2


class DGLOWNetwork(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_levels: int,
        steps_per_level: list[int],
        align_steps_per_level: list[int],
        hidden_channels_per_level: list[int],
        num_resnet_blocks_per_level: list[int],
        block_size_per_level: list[int],
        dropout: float,
        checkpoint_grads: bool = False,
    ):
        super(DGLOWNetwork, self).__init__()
        assert len(steps_per_level) == num_levels, "steps_per_level length must match num_levels"
        assert len(hidden_channels_per_level) == num_levels, "hidden_channels_per_level length must match num_levels"
        assert len(num_resnet_blocks_per_level) == num_levels, "num_resnet_blocks_per_level length must match num_levels"
        
        self.squeeze = Squeeze()
        self.logit_transform = LogitTransform(alpha=0.05)
        self.levels = nn.ModuleList()
        self.num_levels = num_levels
        self.checkpoint_grads = checkpoint_grads
        
        C, H, W = input_shape
        
        for level_idx in range(num_levels):
            C *= 4  # After Squeeze
            H //= 2
            W //= 2
            
            steps = steps_per_level[level_idx]
            align_steps = align_steps_per_level[level_idx]
            hidden_channels = hidden_channels_per_level[level_idx]
            num_blocks = num_resnet_blocks_per_level[level_idx]
            block_size = block_size_per_level[level_idx]

            level_flows = nn.ModuleList()
            for _ in range(steps):
                level_flows.append(FlowStep(
                    in_channels=C, 
                    hidden_channels=hidden_channels,
                    num_resnet_blocks=num_blocks,
                    block_size=block_size,
                    dropout=dropout
                ))
                
            for _ in range(align_steps):
                level_flows.append(FlowStep(
                    in_channels=C, 
                    hidden_channels=hidden_channels,
                    num_resnet_blocks=num_blocks,
                    block_size=2,
                    dropout=dropout
                ))
            
            level_flows.append(Invertible1x1Conv(C))
            self.levels.append(level_flows)
            
            if level_idx < num_levels - 1:
                split = Split(C)
                self.levels.append(split)
                C //= 2

    def forward(self, x):
        log_dets = []
        h, log_det = self.logit_transform(x)
        log_dets.append(log_det)
        
        outs = []
        
        level_log_det = torch.zeros(x.size(0), device=x.device)
        
        for i, level_module in enumerate(self.levels):
            is_flow_level = isinstance(level_module, nn.ModuleList)
            
            if is_flow_level:
                h, log_det_squeeze = self.squeeze(h)
                level_log_det += log_det_squeeze
                
                for flow_step in level_module:
                    if self.checkpoint_grads and h.requires_grad:
                        h, log_det = checkpoint.checkpoint(flow_step, h, use_reentrant=False)
                    else:
                        h, log_det = flow_step(h)
                    level_log_det += log_det

            else: # This is a Split layer
                z, h, log_det_split = level_module(h) # Split h into z (latent) and h (goes to next level)
                outs.append((h, z)) # Append tuple of (h_next, z_current)
                
                level_log_det += log_det_split
                log_dets.append(level_log_det)
                level_log_det = torch.zeros(x.size(0), device=x.device)

        outs.append((None, h))  # Final latent
        log_dets.append(level_log_det)
        
        log_dets = torch.stack(log_dets, dim=0)
        return outs, log_dets
    
    def inverse(self, z_list):
        total_log_det = torch.zeros(z_list[0].shape[0], device=z_list[0].device)
        
        h = z_list.pop() # Start with the final, most semantic latent
        
        for i in range(len(self.levels) - 1, -1, -1):
            level_module = self.levels[i]
            
            if isinstance(level_module, nn.ModuleList): # Flow steps
                for flow_step in reversed(level_module):
                    h, log_det = flow_step.inverse(h)
                    total_log_det += log_det
                    
                h, log_det_squeeze = self.squeeze.inverse(h)
                total_log_det += log_det_squeeze
                
            elif isinstance(level_module, Split): # Split layer
                z = z_list.pop()
                h, log_det_split = level_module.inverse(z, h) # Combine z and h
                total_log_det += log_det_split
                
        x, log_det_logit = self.logit_transform.inverse(h)
        total_log_det += log_det_logit

        return x, total_log_det
    
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