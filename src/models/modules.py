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

    def inverse(self, y, context=None):
        y_a, y_b = y.split([self.split_channels, y.size(1) - self.split_channels], dim=1)
        
        spline_context = self.context_net(y_b, context)
        x_a, log_det = self.block_ar.inverse(y_a, context=spline_context)
        
        x = torch.cat([x_a, y_b], dim=1)
        
        return x, log_det


class ActNorm(nn.Module):
    def __init__(
        self,
        num_channels,
    ):
        super().__init__()
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
            
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


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
        # Initialize with a random orthogonal matrix
        q, _ = torch.linalg.qr(torch.randn(num_channels, num_channels))
        self.weight = nn.Parameter(q)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for matrix multiplication
        x_flat = x.view(B, C, H * W)
        
        # Apply the convolution
        y_flat = self.weight @ x_flat
        
        # Reshape back
        y = y_flat.view(B, C, H, W)
        
        # The log-determinant is H * W * log(abs(det(W)))
        log_det = H * W * torch.slogdet(self.weight)[1]
        log_det = log_det.expand(B) # Expand to batch size
        
        return y, log_det

    def inverse(self, y):
        B, C, H, W = y.shape
        
        # Reshape for matrix multiplication
        y_flat = y.view(B, C, H * W)
        
        # Apply the inverse convolution
        w_inv = torch.inverse(self.weight)
        x_flat = w_inv @ y_flat
        
        # Reshape back
        x = x_flat.view(B, C, H, W)
        
        # The log-determinant of the inverse is the negative of the forward
        log_det = - (H * W * torch.slogdet(self.weight)[1])
        log_det = log_det.expand(B)
        
        return x, log_det


class Split(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.split_point = num_channels // 2

    def forward(self, x):
        # Split the tensor along the channel dimension
        x1, x2 = x.split([self.split_point, x.size(1) - self.split_point], dim=1)
        # The log-determinant is zero for this operation
        return x1, x2, torch.zeros(x.size(0), device=x.device)

    def inverse(self, x1, x2):
        # Concatenate the tensors back together
        x = torch.cat([x1, x2], dim=1)
        # The log-determinant is zero for this operation
        return x, torch.zeros(x.size(0), device=x.device)


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
    def __init__(self, in_channels, hidden_channels, out_channels, num_resnet_blocks=2, dropout=0.0):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList([
            GatedResNetBlock(hidden_channels, dropout=dropout) 
            for _ in range(num_resnet_blocks)
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
    
    def __init__(self, in_channels, hidden_channels=512, num_resnet_blocks=2, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        
        self.coupling_net = CNNCouplingNet(
            self.split_size, 
            hidden_channels, 
            self.in_channels,
            num_resnet_blocks=num_resnet_blocks,
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
            num_resnet_blocks=num_blocks,
            dropout=dropout
        )
        
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
        s_and_t = self.coupling_net(y_a)
        log_s, t = s_and_t.split(self.split_size, dim=1)
        
        scale = self.scale_clamp
        log_s = (2.0 * scale / torch.pi) * torch.atan(log_s / scale)
        s = torch.exp(log_s)
        
        x_b = (y_b - t) / s
        x = torch.cat([y_a, x_b], dim=1)
        log_det = -torch.sum(log_s, dim=[1, 2, 3])
        return x, log_det
    

class MaskedConv2d(nn.Conv2d):
    """
    A 2D convolution with a causal mask on the channels.
    The output for channel `i` depends only on inputs from channels `j <= i` (or `j < i`).
    This is used to build a Masked Autoregressive Flow (MAF).
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        if mask_type not in ['A', 'B']:
            raise ValueError(f"Unknown mask type: {mask_type}")
        self.mask_type = mask_type
        
        mask = torch.ones_like(self.weight.data)
        
        # For each output channel, zero out connections to subsequent input channels
        out_channels_per_group = out_channels // self.groups
        in_channels_per_group = in_channels // self.groups

        for i in range(out_channels_per_group):
            for j in range(in_channels_per_group):
                if self.mask_type == 'A':
                    if j >= i:
                        mask[i::out_channels_per_group, j::in_channels_per_group, :, :] = 0
                else: # 'B'
                    if j > i:
                        mask[i::out_channels_per_group, j::in_channels_per_group, :, :] = 0
        
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Apply the mask before convolution
        self.weight.data *= self.mask
        return super().forward(input)


class MaskedARNet(nn.Module):
    """
    A complete autoregressive network using masked convolutions.
    This replaces the sequential loop of the IAF with a single parallel pass.
    It is "pointwise" because it uses 1x1 convolutions.
    It can be conditioned by adding a projected context tensor.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks=2, dropout=0.0, context_channels=None):
        super().__init__()
        
        # First layer uses mask 'A' to enforce strict autoregression
        self.in_conv = MaskedConv2d(in_channels, hidden_channels, 1, mask_type='A')
        
        # Optional context embedding network
        self.context_embed = None
        if context_channels is not None:
            self.context_embed = nn.Conv2d(context_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # These are not GatedResNetBlocks, but simple residual connections
            # around a sequence of masked convolutions.
            block = nn.Sequential(
                nn.GELU(),
                MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B'),
                nn.GELU(),
                MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B')
            )
            self.blocks.append(block)
            
        # Output layer can use mask 'B'
        self.out_conv = MaskedConv2d(hidden_channels, out_channels, 1, mask_type='B')
        
        # Zero-initialize the final layer for identity at the start
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x, context=None):
        h = self.in_conv(x)
        
        if self.context_embed is not None and context is not None:
            h = h + self.context_embed(context)
            
        for block in self.blocks:
            h = h + block(h) # Residual connection
            
        out = self.out_conv(h)
        return out


class BlockAutoregressiveSpline(nn.Module):
    """
    Splits channels into blocks of size `block_size`.
    Performs autoregressive rational quadratic spline transform WITHIN each block
    using a Masked Autoregressive Flow (MAF) structure for a fast forward pass.
    Weights are shared across blocks (treating blocks as batch dimension).
    This implementation is spatially factorized to ensure a triangular Jacobian.
    """
    def __init__(
        self,
        num_channels,
        context_channels,
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
        self.context_channels = context_channels
        self.block_size = block_size
        self.num_blocks = num_channels // block_size
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        self.params_per_channel = 3 * num_bins + 1
        
        # Total parameters for all splines in a block
        self.total_params = self.block_size * self.params_per_channel

        # The MAF network
        self.ar_net = MaskedARNet(
            in_channels=self.block_size,
            context_channels=self.context_channels,
            hidden_channels=hidden_channels,
            out_channels=self.total_params,
            num_blocks=num_resnet_blocks,
            dropout=dropout
        )
        
        # Initialize derivatives to identity (approx)
        init_val = math.log(math.exp(1 - min_derivative) - 1)
        with torch.no_grad():
            # Reshape bias to (block_size, params_per_channel)
            bias = self.ar_net.out_conv.bias.view(self.block_size, self.params_per_channel)
            
            # Tie heights to widths for identity init
            bias[:, self.num_bins:2*self.num_bins] = bias[:, :self.num_bins]
            
            # Set derivatives
            bias[:, 2*self.num_bins:].fill_(init_val)
            
            self.ar_net.out_conv.bias.data = bias.view(-1)

    def _get_spline_params(self, params_tensor):
        # params_tensor: (B, params_per_channel, H, W)
        # permute to (B, H, W, params_per_channel)
        params_tensor = params_tensor.permute(0, 2, 3, 1)
        unnormalized_widths = params_tensor[..., :self.num_bins]
        unnormalized_heights = params_tensor[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params_tensor[..., 2*self.num_bins:]
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(self, x, context=None):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape to (B * num_blocks, block_size, H, W)
        x_reshaped = x.view(B, self.num_blocks, self.block_size, H, W)
        x_reshaped = x_reshaped.permute(0, 1, 3, 4, 2) # (B, NB, H, W, BS)
        x_reshaped = x_reshaped.reshape(B * self.num_blocks, H, W, self.block_size) # (B*NB, H, W, BS)
        x_reshaped = x_reshaped.permute(0, 3, 1, 2) # (B*NB, BS, H, W)
        
        context_expanded = None
        if context is not None:
            # Expand context to match the reshaped input
            context_expanded = context.repeat_interleave(self.num_blocks, dim=0)
        
        # Get all spline parameters in one parallel pass
        all_params = self.ar_net(x_reshaped, context=context_expanded)
        
        # Reshape params to be (B*NB, block_size, params_per_channel, H, W)
        all_params = all_params.view(
            B * self.num_blocks, self.block_size, self.params_per_channel, H, W
        )
        
        # Permute for spline function: (B*NB, H, W, block_size, params_per_channel)
        all_params = all_params.permute(0, 3, 4, 1, 2)
        
        unnormalized_widths = all_params[..., :self.num_bins]
        unnormalized_heights = all_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = all_params[..., 2*self.num_bins:]
        
        # The input to the spline also needs to be permuted
        # x_reshaped is (B*NB, BS, H, W), we need (B*NB, H, W, BS)
        x_spline_input = x_reshaped.permute(0, 2, 3, 1)
        
        y_spline_output, logabsdet = _rational_quadratic_spline(
            inputs=x_spline_input,
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
        
        # Reshape y back to (B, C, H, W)
        y_spline_output = y_spline_output.permute(0, 3, 1, 2) # (B*NB, BS, H, W)
        y = y_spline_output.contiguous().view(B, C, H, W)
        
        # Sum log determinant
        log_det = logabsdet.view(B, -1).sum(dim=1)
        
        return y, log_det

    def inverse(self, y, context=None):
        # y: (B, C, H, W)
        B, C, H, W = y.shape
            
        # Reshape y to (B * num_blocks, block_size, H, W)
        y_reshaped = y.view(B, self.num_blocks, self.block_size, H, W)
        y_reshaped = y_reshaped.permute(0, 1, 3, 4, 2).contiguous()
        y_reshaped = y_reshaped.view(B * self.num_blocks, H, W, self.block_size)
        y_reshaped = y_reshaped.permute(0, 3, 1, 2)

        # Expand context
        context_expanded = None
        if context is not None:
            context_expanded = context.repeat_interleave(self.num_blocks, dim=0)
        
        # Initialize x with zeros
        x_recon_buffer = torch.zeros_like(y_reshaped)
        
        # The inverse is sequential
        for i in range(self.block_size):
            # Get params for the current channel using previously reconstructed x
            all_params = self.ar_net(x_recon_buffer, context=context_expanded)
            
            # Extract params for the current channel
            params_i = all_params[:, i*self.params_per_channel:(i+1)*self.params_per_channel, :, :]
            params_i = params_i.permute(0, 2, 3, 1) # (B*NB, H, W, params_per_channel)
            
            w = params_i[..., :self.num_bins]
            h = params_i[..., self.num_bins:2*self.num_bins]
            d = params_i[..., 2*self.num_bins:]
            
            # Select the current y channel
            y_i = y_reshaped[:, i, :, :] # (B*NB, H, W)
            
            # Invert the spline
            x_i, _ = _rational_quadratic_spline(
                inputs=y_i,
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
            
            # Store the reconstructed x channel
            x_recon_buffer[:, i, :, :] = x_i
            
        # Re-run the forward pass with the reconstructed x to get the correct log_det
        # This is a common strategy for MAFs to avoid a sequential log_det calculation
        # during the inverse pass.
        x_final = x_recon_buffer.contiguous().view(B, C, H, W)
        _, log_det = self.forward(x_final, context=context)
        
        return x_final, -log_det


class BlockAutoregressiveCoupling(nn.Module):
    """
    A coupling layer that uses a BlockAutoregressiveSpline to transform half of the inputs.
    The other half of the input is used as context to condition the spline transformation.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_resnet_blocks=2,
        block_size=2,
        num_bins=8,
        bound=3.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        
        # The first part of the input is passed through
        self.identity_channels = in_channels // 2
        
        # The second part is transformed
        self.transformed_channels = in_channels - self.identity_channels

        self.spline = BlockAutoregressiveSpline(
            num_channels=self.transformed_channels,
            context_channels=self.identity_channels, # Condition on the other half
            num_resnet_blocks=num_resnet_blocks,
            hidden_channels=hidden_channels,
            block_size=block_size,
            num_bins=num_bins,
            bound=bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            dropout=dropout,
        )

    def forward(self, x):
        x_identity, x_transform = x.split([self.identity_channels, self.transformed_channels], dim=1)
        
        # x_identity is the context for the spline transformation on x_transform
        y_transform, log_det = self.spline(x_transform, context=x_identity)
        
        y = torch.cat([x_identity, y_transform], dim=1)
        
        return y, log_det

    def inverse(self, y):
        y_identity, y_transform = y.split([self.identity_channels, self.transformed_channels], dim=1)
        
        # y_identity is the context for the inverse spline transformation on y_transform
        x_transform, log_det = self.spline.inverse(y_transform, context=y_identity)
        
        x = torch.cat([y_identity, x_transform], dim=1)
        
        return x, log_det
    

if __name__ == '__main__':
    # General parameters
    B, C, H, W = 4, 16, 8, 8
    
    # BlockAutoregressiveSpline parameters
    block_size = 4
    hidden_channels_ar = 32
    num_resnet_blocks = 2
    context_channels_test = 8 # Example context channels
    
    # Test BlockAutoregressiveSpline
    print("--- Testing BlockAutoregressiveSpline (MAF implementation) ---")
    
    # Create inputs
    x = torch.randn(B, C, H, W)
    context = torch.randn(B, context_channels_test, H, W)
    
    # Create model
    try:
        spline_layer = BlockAutoregressiveSpline(
            num_channels=C,
            context_channels=context_channels_test,
            block_size=block_size,
            hidden_channels=hidden_channels_ar,
            num_resnet_blocks=num_resnet_blocks
        )
        
        # Forward pass
        y, log_det_fwd = spline_layer(x, context=context)
        
        # Inverse pass
        x_recon, log_det_inv = spline_layer.inverse(y, context=context)
        
        # --- Checks ---
        # 1. Invertibility check
        inversion_error = torch.abs(x - x_recon).max()
        print(f"Inversion error: {inversion_error.item()}")
        assert torch.allclose(x, x_recon, atol=1e-5), "Inversion failed!"
        
        # 2. Log-determinant check
        log_det_error = torch.abs(log_det_fwd + log_det_inv).mean()
        print(f"Log-determinant error: {log_det_error.item()}")
        assert torch.allclose(log_det_fwd, -log_det_inv, atol=1e-5), "Log-determinant mismatch!"
        
        print("BlockAutoregressiveSpline test PASSED!")
        
    except Exception as e:
        print(f"BlockAutoregressiveSpline test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    # Test BlockAutoregressiveCoupling
    print("--- Testing BlockAutoregressiveCoupling ---")
    
    # Create model
    try:
        coupling_layer = BlockAutoregressiveCoupling(
            in_channels=C,
            hidden_channels=hidden_channels_ar,
            block_size=block_size,
            num_resnet_blocks=num_resnet_blocks
        )
        
        # Forward pass
        y_c, log_det_fwd_c = coupling_layer(x)
        
        # Inverse pass
        x_recon_c, log_det_inv_c = coupling_layer.inverse(y_c)
        
        # --- Checks ---
        # 1. Invertibility check
        inversion_error_c = torch.abs(x - x_recon_c).max()
        print(f"Inversion error: {inversion_error_c.item()}")
        assert torch.allclose(x, x_recon_c, atol=1e-5), "Inversion failed!"
        
        # 2. Log-determinant check
        log_det_error_c = torch.abs(log_det_fwd_c + log_det_inv_c).mean()
        print(f"Log-determinant error: {log_det_error_c.item()}")
        assert torch.allclose(log_det_fwd_c, -log_det_inv_c, atol=1e-5), "Log-determinant mismatch!"
        
        print("BlockAutoregressiveCoupling test PASSED!")
        
    except Exception as e:
        print(f"BlockAutoregressiveCoupling test FAILED: {e}")
        import traceback
        traceback.print_exc()

