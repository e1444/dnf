from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules import Conv2dZeros, Split


def mu_simplex_init(num_classes: int, channels: int, scale: float = 1.0) -> torch.Tensor:
    # Technically should be channels >= num_classes - 1, but it's more convenient this way
    assert channels >= num_classes, "channels must be >= num_classes for simplex initialization"
    mu = torch.zeros((num_classes, channels))
    for k in range(num_classes):
        mu[k, k] = scale
    return mu


class LearnedPrior(nn.Module):
    def __init__(self, shape, init_mu: Optional[torch.Tensor] = None, scale: float = 1.0, cov_method="diag"):
        """
        shape: (C, H, W) - Shape of the latent variable z
        """
        super().__init__()
        self.scale = scale
        self.shape = shape
        self.cov_method = cov_method
        C, H, W = shape
        
        if init_mu is None:
            init_mu = torch.zeros(C)
        
        # Mean is always required
        assert init_mu.shape == (C,), "init_mu must match shape"
        self.mu = nn.Parameter(init_mu.view(C, 1, 1).repeat(1, H, W))
        
        if cov_method == "diag":
            # Diagonal Log-Variance
            self.s = nn.Parameter(torch.zeros(shape))
        elif cov_method == "block_diag":
            # Lower Triangular Parameters for each spatial location
            # Number of params per pixel = C * (C + 1) / 2
            num_cov_params = C * (C + 1) // 2
            self.L_flat = nn.Parameter(torch.zeros(num_cov_params, H, W))
        else:
            raise ValueError(f"Unknown cov_method: {cov_method}")
            
    def forward(self):
        if self.cov_method == "diag":
            logs = self.scale * torch.tanh(self.s)
            constrained_logs = logs - logs.mean(dim=[0,1,2], keepdim=True)
            return self.mu, constrained_logs
        else:
            return self.mu, self.L_flat
    
    
class ConditionalPrior(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_mu: Optional[torch.Tensor] = None, scale: float=1.0, cov_method: str = "diag"):
        super(ConditionalPrior, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self._cov_method = cov_method
        
        if init_mu is None:
            init_mu = torch.zeros(out_channels)
            
        assert init_mu.shape == (out_channels,), "init_mu must match out_channels"
        
        if cov_method == "diag":
            _out_channels = out_channels * 2
            bias = torch.zeros(_out_channels)
            bias[::2] = init_mu.view(-1)
        elif cov_method == "block_diag":
            _out_channels = out_channels * (out_channels + 1) // 2
            _out_channels = out_channels + _out_channels
            bias = torch.zeros(_out_channels)
            bias[:out_channels] = init_mu.view(-1)
        else:
            raise ValueError("cov must be 'diag' or 'block_diag'")
            
        self.conv = Conv2dZeros(in_channels, _out_channels, kernel_size=3, padding=1)
        self.conv.conv.bias.data.copy_(bias)
        self.split = Split()
    
    def forward(self, h: torch.Tensor):
        theta = self.conv(h)
        
        if self._cov_method == "diag":
            mu, s = self.split(theta, method="cross")
            logs = self.scale * torch.tanh(s)
            constrained_logs = logs - logs.mean(dim=[1,2,3], keepdim=True)
            return mu, constrained_logs
        elif self._cov_method == "block_diag":
            mu = theta[:, :self.out_channels, ...]
            L_flat = theta[:, self.out_channels:, ...]
            return mu, L_flat