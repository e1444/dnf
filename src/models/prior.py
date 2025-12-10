import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules import Conv2dZeros, Split


class LearnedPrior(nn.Module):
    def __init__(self, shape, eps=1e-2, cov_method="diag"):
        """
        shape: (C, H, W) - Shape of the latent variable z
        """
        super().__init__()
        self.eps = eps
        self.shape = shape
        self.cov_method = cov_method
        C, H, W = shape
        
        # Mean is always required
        self.mu = nn.Parameter(torch.zeros(shape))
        
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
            sigma = F.softplus(self.s) + self.eps
            self.logs = torch.log(sigma)
            constrained_logs = self.logs - self.logs.mean(dim=[0,1,2], keepdim=True)
            return self.mu, constrained_logs
        else:
            return self.mu, self.L_flat
    
    
class ConditionalPrior(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps=1e-2, cov_method: str = "diag"):
        super(ConditionalPrior, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self._cov_method = cov_method
        
        if cov_method == "diag":
            _out_channels = out_channels * 2
        elif cov_method == "block_diag":
            _out_channels = out_channels * (out_channels + 1) // 2
            _out_channels = out_channels + _out_channels
        else:
            raise ValueError("cov must be 'diag' or 'block_diag'")
            
        self.conv = Conv2dZeros(in_channels, _out_channels, kernel_size=3, padding=1)
        self.split = Split()
    
    def forward(self, h: torch.Tensor):
        theta = self.conv(h)
        
        if self._cov_method == "diag":
            mu, s = self.split(theta, method="cross")
            sigma = F.softplus(s) + self.eps
            logs = torch.log(sigma)
            constrained_logs = logs - logs.mean(dim=[1,2,3], keepdim=True)
            return mu, constrained_logs
        elif self._cov_method == "block_diag":
            mu = theta[:, :self.out_channels, ...]
            L_flat = theta[:, self.out_channels:, ...]
            return mu, L_flat