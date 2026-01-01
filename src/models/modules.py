import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as la

import math
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Performs attention-based pooling. For each channel, it learns a spatial
    attention map to compute a weighted average of features.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.attention_conv = nn.Conv2d(num_features, num_features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Pooled feature vector of shape (B, C)
        """
        attention_logits = self.attention_conv(x) # (B, C, H, W)
        
        B, C, H, W = attention_logits.shape
        attention_logits_flat = attention_logits.view(B, C, H * W)
        attention_weights = nn.functional.softmax(attention_logits_flat, dim=-1) # (B, C, H*W)
        
        x_flat = x.view(B, C, H * W)
        pooled_features = torch.sum(x_flat * attention_weights, dim=-1) # (B, C)
        
        return pooled_features
    
    
class GaussianBlurLayer(nn.Module):
    """
    Applies a fixed Gaussian blur using a grouped 2D convolution.
    This is efficient, differentiable, and GPU-friendly.
    """
    weight: torch.Tensor
    
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.padding = kernel_size // 2
        self.groups = channels
        
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
        
        self.register_buffer('weight', kernel_2d)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)
    

class LogitTransform(nn.Module):
    """
    Maps data from [0, 1] to (-inf, inf) using logit(alpha + (1 - 2*alpha) * x).
    Standard preprocessing for images in Flows.
    """
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # x in [0, 1]
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        
        log_det = torch.sum(
            torch.log(torch.tensor(1 - 2 * self.alpha, device=x.device))
            - torch.log(s) - torch.log(1 - s),
            dim=[1, 2, 3]
        )
        return y, log_det

    def inverse(self, y):
        s = torch.sigmoid(y)
        x = (s - self.alpha) / (1 - 2 * self.alpha)
        
        log_det = -torch.sum(
            torch.log(torch.tensor(1 - 2 * self.alpha, device=y.device))
            - torch.log(s) - torch.log(1 - s),
            dim=[1, 2, 3]
        )
        return x, log_det


class ActNorm(nn.Module):
    def __init__(
        self,
        num_channels,
        initialization="identity"
    ):
        super().__init__()
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        if initialization == "data-dependent":
            self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        elif initialization == "identity":
            self.register_buffer("initialized", torch.tensor(1, dtype=torch.uint8))
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
            
    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                mean = x.mean(dim=[0, 2, 3], keepdim=True)
                std = x.std(dim=[0, 2, 3], keepdim=True)
                self.logs.data.copy_(-torch.log(std + 1e-6))
                self.bias.data.copy_(-mean * torch.exp(self.logs))
                self.initialized.fill_(1)

        y = torch.exp(self.logs) * x + self.bias
        _, _, h, w = x.size()
        log_det = torch.sum(self.logs) * h * w
        return y, log_det

    def inverse(self, y):
        x = (y - self.bias) * torch.exp(-self.logs)
        _, _, h, w = y.size()
        log_det = -torch.sum(self.logs) * h * w
        return x, log_det


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        log_det = torch.zeros(b, device=x.device)
        return x, log_det
    
    def inverse(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
        log_det = torch.zeros(b, device=x.device)
        return x, log_det


class GatedConv2d(nn.Module):
    """
    Combines Conv2d and Gated Activation.
    Flow++ uses: Conv -> Split -> a * sigmoid(b)
    Your stable version: Conv -> Split -> tanh(a) * sigmoid(b)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.conv(x)
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class GatedResNetBlock(nn.Module):
    """
    Flow++ Residual Block:
    x -> Conv(1x1) -> GatedAct -> Conv(3x3) -> GatedAct -> Conv(1x1) -> Dropout -> + x
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, kernel_size=1, padding=0)
        self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = GatedConv2d(channels, channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)
        return residual + out


class CNNCouplingNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks=2, dropout=0.0):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList([
            GatedResNetBlock(hidden_channels, dropout=dropout) 
            for _ in range(num_blocks)
        ])
        
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        
        # Zero initialization for the last layer (Identity Init)
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()     # type: ignore

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.out_conv(x)


class AffineCoupling(nn.Module):
    scale_clamp: torch.Tensor
    
    def __init__(self, in_channels, hidden_channels=512, num_blocks=2, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        
        self.coupling_net = CNNCouplingNet(
            self.split_size, 
            hidden_channels, 
            self.in_channels,
            num_blocks=num_blocks,
            dropout=dropout
        )
        
        self.register_buffer("scale_clamp", torch.tensor(1.0))


    def forward(self, x):
        x_a, x_b = x.split(self.split_size, dim=1)
        s_and_t = self.coupling_net(x_a)
        log_s, t = s_and_t.split(self.split_size, dim=1)
        
        scale = self.scale_clamp
        log_s = (2.0 * scale / torch.pi) * torch.atan(log_s / scale)
        s = torch.exp(log_s)

        y_b = x_b * s + t
        y = torch.cat([x_a, y_b], dim=1)
        log_det = torch.sum(torch.log(s), dim=[1, 2, 3])
        return y, log_det

    def inverse(self, y):
        y_a, y_b = y.split(self.split_size, dim=1)
        s_and_t = self.coupling_net(y_a)
        log_s, t = s_and_t.split(self.split_size, dim=1)
        
        scale = self.scale_clamp
        log_s = (2.0 * scale / torch.pi) * torch.atan(log_s / scale)
        s = torch.exp(log_s)
        
        x_b = (y_b - t) / s
        x = torch.cat([y_a, x_b], dim=1)
        log_det = -torch.sum(log_s, dim=[1, 2, 3])
        return x, log_det
    

def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=-3.0,
    right=3.0,
    bottom=-3.0,
    top=3.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    Rational quadratic spline transformation.
    Based on: https://github.com/bayesiains/nsf/blob/master/utils/rational_quadratic_spline.py
    """
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Handle out-of-bounds values with linear tails (identity)
    # We clamp inputs to the range, apply spline, and add back the residual
    inputs_clamped = torch.clamp(inputs, left, right)
    
    if inverse:
        bin_idx = torch.searchsorted(cumheights, inputs_clamped[..., None].contiguous()) - 1
        bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)
        
        input_cumwidths = cumheights
        output_cumwidths = cumwidths
        input_bin_widths = heights
        output_bin_widths = widths
        input_derivatives = derivatives
    else:
        bin_idx = torch.searchsorted(cumwidths, inputs_clamped[..., None].contiguous()) - 1
        bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)
        
        input_cumwidths = cumwidths
        output_cumwidths = cumheights
        input_bin_widths = widths
        output_bin_widths = heights
        input_derivatives = derivatives

    input_cumwidths = input_cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = input_bin_widths.gather(-1, bin_idx)[..., 0]

    output_cumwidths = output_cumwidths.gather(-1, bin_idx)[..., 0]
    output_bin_widths = output_bin_widths.gather(-1, bin_idx)[..., 0]

    input_derivatives_plus_one = input_derivatives[..., 1:]
    input_derivatives = input_derivatives[..., :-1]

    input_derivatives = input_derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = input_derivatives_plus_one.gather(-1, bin_idx)[..., 0]

    # s = h / w
    if inverse:
        s = input_bin_widths / output_bin_widths
        
        # Solve quadratic for theta (xi)
        y_rel = inputs_clamped - input_cumwidths
        w = output_bin_widths
        h = input_bin_widths
        d0 = input_derivatives
        d1 = input_derivatives_plus_one
        sum_d = d0 + d1
        term = sum_d - 2 * s
        
        a = h * (s - d0) + y_rel * term
        b = h * d0 - y_rel * term
        c = -s * y_rel
        
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2a
        delta = b.pow(2) - 4 * a * c
        
        # Avoid division by zero when a is small (linear part of spline)
        mask = torch.abs(a) > 1e-6
        numerator = -b + torch.sqrt(delta)
        denominator = 2 * a
        
        theta = torch.where(mask, numerator / denominator, -c / b)
        
        # Calculate outputs (x)
        theta_one_minus_theta = theta * (1 - theta)
        outputs = output_cumwidths + theta * output_bin_widths
        
        # Calculate derivative for logabsdet
        derivative_numerator = s.pow(2) * (
            d1 * theta.pow(2)
            + 2 * s * theta_one_minus_theta
            + d0 * (1 - theta).pow(2)
        )
        denominator = s + term * theta_one_minus_theta
        logabsdet = 2 * torch.log(denominator) - torch.log(derivative_numerator)
        
    else:
        s = output_bin_widths / input_bin_widths
        theta = (inputs_clamped - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = output_bin_widths * (
            s * theta * theta
            + input_derivatives * theta_one_minus_theta
        )
        denominator = s + (
            input_derivatives + input_derivatives_plus_one - 2 * s
        ) * theta_one_minus_theta
        outputs = output_cumwidths + numerator / denominator

        derivative_numerator = s.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * s * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    # Add back the linear tail (identity)
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outputs = outputs * inside_interval_mask.float() + inputs * (~inside_interval_mask).float()
    logabsdet = logabsdet * inside_interval_mask.float()

    return outputs, logabsdet


class PiecewiseRationalQuadraticCoupling(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        num_blocks=2,
        num_bins=8,
        dropout=0.0,
        bound=3.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Output dimension: (3 * num_bins + 1) parameters per channel
        # - num_bins widths
        # - num_bins heights
        # - num_bins + 1 derivatives
        out_dim = self.split_size * (3 * num_bins + 1)
        
        self.coupling_net = CNNCouplingNet(
            self.split_size,
            hidden_channels,
            out_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
        
        # Initialize last layer to zero (already done in CNNCouplingNet)
        # BUT we need to set the bias for derivatives to give identity
        # softplus(bias) + min_derivative = 1
        # softplus(bias) = 1 - min_derivative
        # bias = inverse_softplus(1 - min_derivative)
        # bias = log(exp(1 - min_derivative) - 1)
        
        init_val = math.log(math.exp(1 - min_derivative) - 1)
        
        with torch.no_grad():
            bias = self.coupling_net.out_conv.bias.view(self.split_size, -1) # (split_size, 3*num_bins+1)
            
            # Tie heights to widths at init for exact identity
            bias[:, self.num_bins:2*self.num_bins] = bias[:, :self.num_bins]
            
            # Derivatives
            bias[:, 2*self.num_bins:] = init_val
            self.coupling_net.out_conv.bias.data = bias.view(-1)

    def forward(self, x):
        x_a, x_b = x.split(self.split_size, dim=1)
        
        params = self.coupling_net(x_a)
        
        B, _, H, W = params.shape
        params = params.view(B, self.split_size, -1, H, W).permute(0, 1, 3, 4, 2)
        
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:]
        
        outputs, logabsdet = _rational_quadratic_spline(
            inputs=x_b,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=False,
            left=-self.bound, right=self.bound,
            bottom=-self.bound, top=self.bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = torch.cat([x_a, outputs], dim=1)
        log_det = torch.sum(logabsdet, dim=[1, 2, 3])
        return y, log_det

    def inverse(self, y):
        y_a, y_b = y.split(self.split_size, dim=1)
        
        params = self.coupling_net(y_a)
        B, _, H, W = params.shape
        params = params.view(B, self.split_size, -1, H, W).permute(0, 1, 3, 4, 2)
        
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:]
        
        outputs, logabsdet = _rational_quadratic_spline(
            inputs=y_b,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=True,
            left=-self.bound, right=self.bound,
            bottom=-self.bound, top=self.bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        x = torch.cat([y_a, outputs], dim=1)
        log_det = torch.sum(logabsdet, dim=[1, 2, 3])
        return x, log_det
    

class BlockAutoregressiveSpline(nn.Module):
    """
    Splits channels into blocks of size `block_size`.
    Performs autoregressive rational quadratic spline transform WITHIN each block.
    Weights are shared across blocks (treating blocks as batch dimension).
    
    This allows for "weakly multi-dimensional" splines that can rotate/mix 
    local groups of channels without the O(bins^k) cost of full multi-dim splines.
    """
    def __init__(
        self,
        num_channels,
        num_resnet_blocks=1,
        hidden_channels=64,
        block_size=2,
        num_bins=8,
        bound=3.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        dropout=0.0,
    ):
        super().__init__()
        if num_channels % block_size != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by block_size ({block_size})")
            
        self.num_channels = num_channels
        self.block_size = block_size
        self.num_blocks = num_channels // block_size
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        # Scale hidden_channels to avoid memory explosion
        # We aim to keep total activations roughly constant:
        # num_blocks * local_hidden ~= global_hidden
        # We enforce a minimum of 32 channels to ensure expressivity.
        self.hidden_channels = min(hidden_channels, max(hidden_channels // self.num_blocks, 32))
        
        # Output dim for one channel's spline params
        self.params_per_channel = 3 * num_bins + 1
        
        # Parameters for the first channel in the block (unconditioned)
        self.first_channel_params = nn.Parameter(torch.zeros(1, self.params_per_channel, 1, 1))
        
        # Networks for subsequent channels
        self.ar_nets = nn.ModuleList()
        for i in range(1, block_size):
            net = CNNCouplingNet(
                in_channels=i,
                hidden_channels=self.hidden_channels,
                out_channels=self.params_per_channel,
                num_blocks=num_resnet_blocks,
                dropout=dropout
            )
            
            # Initialize derivatives to identity (approx)
            init_val = math.log(math.exp(1 - min_derivative) - 1)
            with torch.no_grad():
                net.out_conv.bias[-self.num_bins-1:].fill_(init_val)
                
            self.ar_nets.append(net)
            
        with torch.no_grad():
            self.first_channel_params.data[:, -self.num_bins-1:, :, :].fill_(init_val)


    def _get_spline_params(self, params_tensor):
        # params_tensor: (B, params_per_channel, H, W)
        # permute to (B, H, W, params_per_channel)
        params_tensor = params_tensor.permute(0, 2, 3, 1)
        unnormalized_widths = params_tensor[..., :self.num_bins]
        unnormalized_heights = params_tensor[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params_tensor[..., 2*self.num_bins:]
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        x_reshaped = x.view(B, self.num_blocks, self.block_size, H, W)
        x_reshaped = x_reshaped.view(B * self.num_blocks, self.block_size, H, W)
        
        outputs_list = []
        total_log_det = 0
        
        for i in range(self.block_size):
            curr_x = x_reshaped[:, i] # (B*NB, H, W)
            
            if i == 0:
                params = self.first_channel_params.expand(B * self.num_blocks, -1, H, W)
            else:
                context = x_reshaped[:, :i] # (B*NB, i, H, W)
                params = self.ar_nets[i-1](context)
            
            w, h, d = self._get_spline_params(params)
            
            curr_y, log_det = _rational_quadratic_spline(
                inputs=curr_x,
                unnormalized_widths=w,
                unnormalized_heights=h,
                unnormalized_derivatives=d,
                inverse=False,
                left=-self.bound, right=self.bound,
                bottom=-self.bound, top=self.bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            
            outputs_list.append(curr_y)
            total_log_det = total_log_det + log_det.sum(dim=[1, 2]) # Sum over H, W
            
        y_reshaped = torch.stack(outputs_list, dim=1)
        y = y_reshaped.view(B, self.num_blocks, self.block_size, H, W).view(B, C, H, W)
        
        log_det = total_log_det.view(B, self.num_blocks).sum(dim=1)
        
        return y, log_det

    def inverse(self, y):
        # y: (B, C, H, W)
        B, C, H, W = y.shape
        
        # Reshape to (B * num_blocks, block_size, H, W)
        y_reshaped = y.view(B, self.num_blocks, self.block_size, H, W)
        y_reshaped = y_reshaped.view(B * self.num_blocks, self.block_size, H, W)
        
        # We need to reconstruct x sequentially
        # Temporary buffer for reconstructed x
        x_recon_buffer = torch.zeros_like(y_reshaped)
        total_log_det = 0
        
        for i in range(self.block_size):
            curr_y = y_reshaped[:, i]
            
            if i == 0:
                params = self.first_channel_params.expand(B * self.num_blocks, -1, H, W)
            else:
                # Condition on PREVIOUSLY RECONSTRUCTED x
                context = x_recon_buffer[:, :i]
                params = self.ar_nets[i-1](context)
                
            w, h, d = self._get_spline_params(params)
            
            curr_x, log_det = _rational_quadratic_spline(
                inputs=curr_y,
                unnormalized_widths=w,
                unnormalized_heights=h,
                unnormalized_derivatives=d,
                inverse=True,
                left=-self.bound, right=self.bound,
                bottom=-self.bound, top=self.bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            
            x_recon_buffer[:, i] = curr_x
            total_log_det = total_log_det + log_det.sum(dim=[1, 2])

        x = x_recon_buffer.view(B, self.num_blocks, self.block_size, H, W).view(B, C, H, W)
        log_det = total_log_det.view(B, self.num_blocks).sum(dim=1)
        
        return x, log_det
    

class Invertible1x1Conv(nn.Module):
    p: torch.Tensor
    sign_s: torch.Tensor
    l_mask: torch.Tensor
    u_mask: torch.Tensor
    
    def __init__(self, num_channels, initialization="orthogonal"):
        super().__init__()
        w_shape = [num_channels, num_channels]
        
        if initialization == "orthogonal":
            w_init = torch.linalg.qr(torch.randn(*w_shape))[0]
        elif initialization == "identity":
            w_init = torch.eye(num_channels)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        P, L, U = torch.linalg.lu(w_init)
        
        s = torch.diag(U)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        U = torch.triu(U, diagonal=1)
        
        self.register_buffer("p", P)
        self.register_buffer("sign_s", sign_s)
        self.l = nn.Parameter(L)
        self.log_s = nn.Parameter(log_s)
        self.u = nn.Parameter(U)
        
        self.register_buffer("l_mask", torch.tril(torch.ones(w_shape), diagonal=-1))
        self.register_buffer("u_mask", torch.triu(torch.ones(w_shape), diagonal=1))

    def get_weight(self):
        l = self.l * self.l_mask + torch.eye(self.l.size(0), device=self.l.device)
        u = self.u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        w = torch.matmul(self.p, torch.matmul(l, u))
        return w.view(w.size(0), w.size(1), 1, 1)

    def forward(self, x):
        w = self.get_weight()
        y = nn.functional.conv2d(x, w)
        _, _, h, w_dim = x.size()
        log_det = torch.sum(self.log_s) * h * w_dim
        return y, log_det

    def inverse(self, y):
        w = self.get_weight()
        w_inv = torch.linalg.inv(w.squeeze()).view(w.size(0), w.size(1), 1, 1)
        x = nn.functional.conv2d(y, w_inv)
        _, _, h, w_dim = y.size()
        log_det = -torch.sum(self.log_s) * h * w_dim
        return x, log_det


class Split(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1, x2

    def inverse(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x, 0


if __name__ == "__main__":
    import math
    
    print("Running sanity checks for PiecewiseRationalQuadraticCoupling...")
    
    C, H, W = 4, 32, 32
    coupling = PiecewiseRationalQuadraticCoupling(in_channels=C, hidden_channels=64, num_bins=4)
    
    # 1. Check Identity Initialization
    x = torch.randn(2, C, H, W)
    with torch.no_grad():
        y, log_det = coupling(x)
        
    diff = (x - y).abs().max().item()
    log_det_max = log_det.abs().max().item()
    
    print(f"Identity Init Check:")
    print(f"  Max diff (x - y): {diff:.6e}")
    print(f"  Max log_det: {log_det_max:.6e}")
    
    if diff < 2e-4 and log_det_max < 2e-4:
        print("  [PASS] Identity initialization looks correct.")
    else:
        print("  [FAIL] Identity initialization failed.")

    # 2. Check Invertibility
    # Perturb weights to make it non-identity
    with torch.no_grad():
        coupling.coupling_net.out_conv.weight.data.normal_(0, 0.01)
        
    with torch.no_grad():
        y, log_det_fwd = coupling(x)
        x_recon, log_det_inv = coupling.inverse(y)
        
    recon_diff = (x - x_recon).abs().max().item()
    log_det_diff = (log_det_fwd + log_det_inv).abs().max().item()
    
    print(f"\nInvertibility Check (Random Weights):")
    print(f"  Max diff (x - x_recon): {recon_diff:.6e}")
    print(f"  Max log_det sum: {log_det_diff:.6e}")
    
    if recon_diff < 1e-3 and log_det_diff < 1e-3:
        print("  [PASS] Invertibility looks correct.")
    else:
        print("  [FAIL] Invertibility failed.")
        
        print("\nRunning sanity checks for BlockAutoregressiveSpline...")
    
    C, H, W = 4, 16, 16
    block_size = 2
    ar_spline = BlockAutoregressiveSpline(num_channels=C, block_size=block_size, hidden_channels=32, num_bins=4)
    
    # 1. Check Identity Initialization
    x = torch.randn(2, C, H, W)
    with torch.no_grad():
        y, log_det = ar_spline(x)
        
    diff = (x - y).abs().max().item()
    log_det_max = log_det.abs().max().item()
    
    print(f"Identity Init Check:")
    print(f"  Max diff (x - y): {diff:.6e}")
    print(f"  Max log_det: {log_det_max:.6e}")
    
    if diff < 2e-4 and log_det_max < 2e-4:
        print("  [PASS] Identity initialization looks correct.")
    else:
        print("  [FAIL] Identity initialization failed.")

    # 2. Check Invertibility
    # Perturb weights
    with torch.no_grad():
        ar_spline.first_channel_params.data.normal_(0, 0.1)
        for net in ar_spline.ar_nets:
            for p in net.parameters():
                p.data.normal_(0, 0.01)
                
    with torch.no_grad():
        y, log_det_fwd = ar_spline(x)
        x_recon, log_det_inv = ar_spline.inverse(y)
        
    recon_diff = (x - x_recon).abs().max().item()
    log_det_diff = (log_det_fwd + log_det_inv).abs().max().item()
    
    print(f"\nInvertibility Check (Random Weights):")
    print(f"  Max diff (x - x_recon): {recon_diff:.6e}")
    print(f"  Max log_det sum: {log_det_diff:.6e}")
    
    if recon_diff < 1e-3 and log_det_diff < 1e-3:
        print("  [PASS] Invertibility looks correct.")
    else:
        print("  [FAIL] Invertibility failed.")