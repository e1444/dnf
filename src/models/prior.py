import torch
import torch.nn as nn
from src.models.modules import Conv2dZeros, Split


class LearnedPrior(nn.Module):
    def __init__(self, shape, cov_method="diag"):
        """
        shape: (C, H, W) - Shape of the latent variable z
        """
        super().__init__()
        self.shape = shape
        self.cov_method = cov_method
        C, H, W = shape
        
        # Mean is always required
        self.mu = nn.Parameter(torch.zeros(shape))
        
        if cov_method == "diag":
            # Diagonal Log-Variance
            self.logs = nn.Parameter(torch.zeros(shape))
        elif cov_method == "block_diag":
            # Lower Triangular Parameters for each spatial location
            # Number of params per pixel = C * (C + 1) / 2
            num_cov_params = C * (C + 1) // 2
            self.L_flat = nn.Parameter(torch.zeros(num_cov_params, H, W))
            
            # Initialize diagonal of L to 1 (log-diagonal to 0) for identity covariance
            # We need to figure out which indices correspond to the diagonal
            # But since we init with zeros, L will be zero matrix. 
            # We usually want L to be Identity (or close to it).
            # A common trick is to add Identity to the constructed L matrix later,
            # or initialize the diagonal parts of L_flat to a value that gives 1.
        else:
            raise ValueError(f"Unknown cov_method: {cov_method}")
            
    def forward(self):
        if self.cov_method == "diag":
            constrained_logs = self.logs - self.logs.mean()
            return self.mu, constrained_logs
        else:
            return self.mu, self.L_flat
    
    
class ConditionalPrior(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cov_method: str = "diag"):
        super(ConditionalPrior, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
            mu, logs = self.split(theta, method="cross")
            constrained_logs = logs - logs.mean()
            return mu, constrained_logs
        elif self._cov_method == "block_diag":
            mu = theta[:, :self.out_channels, ...]
            L_flat = theta[:, self.out_channels:, ...]
            return mu, L_flat