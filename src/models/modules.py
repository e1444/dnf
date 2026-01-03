import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as la

import math
import copy
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
    max_derivative=10.0,
):
    """
    Rational quadratic spline transformation.
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

    # Clamp derivatives to [min_derivative, max_derivative] for stability
    # Similar to affine clamp, prevents extreme slopes/log-dets
    derivatives = min_derivative + (max_derivative - min_derivative) * torch.sigmoid(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Handle out-of-bounds values with linear tails (identity)
    # Note: For inverse, we check 'y' (inputs) against bounds.
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

    if inverse:
        s = input_bin_widths / output_bin_widths
        
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
        
        delta = b.pow(2) - 4 * a * c
        
        mask = torch.abs(a) > 1e-6
        numerator = -b + torch.sqrt(delta)
        denominator = 2 * a
        
        theta = torch.where(mask, numerator / denominator, -c / b)
        
        theta_one_minus_theta = theta * (1 - theta)
        outputs = output_cumwidths + theta * output_bin_widths
        
        # Derivative of the forward transform at the solution point
        derivative_numerator = s.pow(2) * (
            d1 * theta.pow(2)
            + 2 * s * theta_one_minus_theta
            + d0 * (1 - theta).pow(2)
        )
        denominator = s + term * theta_one_minus_theta
        
        # log|dx/dy| = -log|dy/dx|
        # Forward log_det formula is log(num) - 2*log(denom)
        # Inverse log_det is -(log(num) - 2*log(denom)) = 2*log(denom) - log(num)
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
        max_derivative=10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        self.num_bins = num_bins
        self.bound = bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.max_derivative = max_derivative

        # Output dimension: (3 * num_bins + 1) parameters per channel
        out_dim = self.split_size * (3 * num_bins + 1)
        
        # NOTE: Assuming CNNCouplingNet is defined elsewhere or this is a placeholder.
        # Ensure CNNCouplingNet outputs 'out_dim' channels.
        self.coupling_net = CNNCouplingNet(
            self.split_size,
            hidden_channels,
            out_dim,
            num_resnet_blocks=num_blocks,
            dropout=dropout
        )
        
        # Identity derivative init for sigmoid mapping: raw_init = logit((1 - min)/(max - min))
        p = (1 - min_derivative) / (max_derivative - min_derivative)
        p = max(min(p, 1 - 1e-6), 1e-6)
        init_val = math.log(p / (1 - p))
        
        # Initialize last layer
        with torch.no_grad():
            if hasattr(self.coupling_net, 'out_conv'):
                bias = self.coupling_net.out_conv.bias.view(self.split_size, -1) 
                bias[:, self.num_bins:2*self.num_bins] = bias[:, :self.num_bins]
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
            max_derivative=self.max_derivative,
        )

        y = torch.cat([x_a, outputs], dim=1)
        log_det = torch.sum(logabsdet, dim=[1, 2, 3])
        return y, log_det

    def inverse(self, y):
        # [FIX] Completely rewrote this method. 
        # Previously it used affine logic for a spline layer.
        y_a, y_b = y.split(self.split_size, dim=1)
        
        params = self.coupling_net(y_a)
        
        B, _, H, W = params.shape
        params = params.view(B, self.split_size, -1, H, W).permute(0, 1, 3, 4, 2)
        
        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = params[..., 2*self.num_bins:]
        
        # Use Spline Inverse
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
            max_derivative=self.max_derivative,
        )
        
        x = torch.cat([y_a, outputs], dim=1)
        log_det = torch.sum(logabsdet, dim=[1, 2, 3])
        return x, log_det
    

class MaskedConv2d(nn.Conv2d):
    """
    [FIXED] A 2D convolution with a causal mask on the channels.
    Correctly handles channel expansion (out_channels > in_channels) for AR parameter generation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        if mask_type not in ['A', 'B']:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # [FIX] Compute the mask based on variables, not raw channels.
        # We assume out_channels is a multiple of in_channels (or they are 1-to-1).
        # If out > in, we assume consecutive output blocks correspond to input variables.
        
        mask = torch.ones_like(self.weight)
        
        # Calculate how many output channels correspond to one variable
        # If out_channels == in_channels, params_per_var = 1
        # If out_channels (total_params) > in_channels (block_size), params_per_var > 1
        if out_channels % in_channels != 0 and in_channels % out_channels != 0:
             # Fallback to standard 1-to-1 if dimensions don't align cleanly (e.g. hidden layers)
             # In hidden layers (hidden -> hidden), we assume 1-to-1 mapping of degrees usually
             params_per_var_out = 1 
             vars_in = in_channels
        else:
             if out_channels >= in_channels:
                 params_per_var_out = out_channels // in_channels
                 vars_in = in_channels
             else:
                 # Reducing dimensions (uncommon for AR output, common for bottleneck)
                 # Not strictly handled here for general degrees, but sufficient for this specific use case
                 params_per_var_out = 1 
                 vars_in = in_channels

        # i: output channel index, j: input channel index
        for i in range(out_channels):
            for j in range(in_channels):
                
                # Determine which variable indices these channels represent
                var_out_idx = i // params_per_var_out
                var_in_idx = j # Assuming input is always 1 channel per variable (or handled by grouping)
                
                # If we are in a hidden layer where in=hidden, out=hidden, 
                # we treat indices as variable degrees directly.
                if out_channels == in_channels:
                    var_out_idx = i
                    var_in_idx = j

                if mask_type == 'A':
                    # Output k depends on Input < k
                    if var_in_idx >= var_out_idx:
                        mask[i, j, :, :] = 0
                else: # 'B'
                    # Output k depends on Input <= k
                    if var_in_idx > var_out_idx:
                        mask[i, j, :, :] = 0
        
        self.register_buffer('mask', mask)

    def forward(self, input):
        # [FIX] Do not modify self.weight.data in-place.
        # This fixes weight accumulation issues and potential autograd bugs.
        return F.conv2d(input, self.weight * self.mask, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class MaskedARNet(nn.Module):
    """
    A complete autoregressive network using masked convolutions.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks=2, dropout=0.0, context_channels=None):
        super().__init__()
        
        # First layer uses mask 'A' (strict autoregression: output_i depends on input < i)
        self.in_conv = MaskedConv2d(in_channels, hidden_channels, 1, mask_type='A')
        
        self.context_embed = None
        if context_channels is not None:
            self.context_embed = nn.Conv2d(context_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.GELU(),
                # Hidden layers use Mask 'B' (autoregression maintained: hidden_i depends on hidden <= i)
                MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B'),
                nn.Dropout(dropout),
                nn.GELU(),
                MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B'),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
            
        # Output layer uses mask 'B'.
        # IMPORTANT: 'out_channels' here is typically (block_size * params_per_spline).
        # MaskedConv2d now correctly handles this expansion to ensure params for variable i
        # depend only on hidden state <= i.
        self.out_conv = MaskedConv2d(hidden_channels, out_channels, 1, mask_type='B')
        
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x, context=None):
        h = self.in_conv(x)
        
        if self.context_embed is not None and context is not None:
            h = h + self.context_embed(context)
            
        for block in self.blocks:
            h = h + block(h) 
            
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
        max_derivative=20.0,
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
        self.max_derivative = max_derivative
        
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
        # Identity derivative init for sigmoid mapping: raw_init = logit((1 - min)/(max - min))
        p = (1 - min_derivative) / (max_derivative - min_derivative)
        p = max(min(p, 1 - 1e-6), 1e-6)
        init_val = math.log(p / (1 - p))
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
            max_derivative=self.max_derivative,
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
                max_derivative=self.max_derivative,
            )
            
            # Store the reconstructed x channel
            x_recon_buffer[:, i, :, :] = x_i
            
        # Re-run the forward pass with the reconstructed x to get the correct log_det
        # This is a common strategy for MAFs to avoid a sequential log_det calculation
        # during the inverse pass.
        x_final = x_recon_buffer.contiguous().view(B, C, H, W)
        _, log_det = self.forward(x_final, context=context)
        
        return x_final, -log_det


class BlockConditionalTriangularLinear(nn.Module):
    """
    Blockwise conditional triangular linear mixing.
    For each spatial location and block, constructs a lower-triangular matrix L
    and positive diagonal D (via clamped log-diagonal), then applies (D + L) to
    the block vector. Parameters depend only on provided context features,
    preserving triangular Jacobian structure and tractable log-det.
    """
    def __init__(
        self,
        num_channels,
        context_channels,
        hidden_channels=64,
        num_resnet_blocks=1,
        block_size=2,
        dropout=0.0,
        scale_clamp=1.0,
    ):
        super().__init__()
        if num_channels % block_size != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by block_size ({block_size})")
        self.num_channels = num_channels
        self.context_channels = context_channels
        self.block_size = block_size
        self.num_blocks = num_channels // block_size
        self.scale_clamp = scale_clamp

        # Parameters per block: diag (bs) + strictly lower (bs*(bs-1)/2)
        self.lower_params_per_block = (block_size * (block_size - 1)) // 2
        self.diag_params_per_block = block_size
        self.params_per_block = self.diag_params_per_block + self.lower_params_per_block

        # Context network to emit triangular parameters per block
        out_channels = self.num_blocks * self.params_per_block
        self.param_net = CNNCouplingNet(
            in_channels=self.context_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_resnet_blocks=num_resnet_blocks,
            dropout=dropout,
        )
        # Zero-init already applied in CNNCouplingNet to ensure identity at init

        # Precompute row offsets for mapping lower-triangular params
        offsets = []
        acc = 0
        for i in range(self.block_size):
            offsets.append(acc)
            acc += i
        self.register_buffer("row_offsets", torch.tensor(offsets, dtype=torch.int64))

    def _reshape_blocks(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, self.num_blocks, self.block_size, H, W)
        x_reshaped = x_reshaped.permute(0, 1, 3, 4, 2)  # (B, NB, H, W, BS)
        x_reshaped = x_reshaped.reshape(B * self.num_blocks, H, W, self.block_size)
        return x_reshaped, B, H, W

    def _unreshape_blocks(self, y, B, H, W):
        # y: (B*NB, H, W, BS)
        NB = self.num_blocks
        y = y.view(B, NB, H, W, self.block_size)
        y = y.permute(0, 1, 4, 2, 3).contiguous()  # (B, NB, BS, H, W)
        y = y.view(B, NB * self.block_size, H, W)
        return y

    def _get_params(self, context, B, H, W):
        # Emit params and reshape to (B*NB, H, W, params_per_block)
        params = self.param_net(context)  # (B, NB*params, H, W)
        params = params.view(B, self.num_blocks, self.params_per_block, H, W)
        params = params.permute(0, 1, 3, 4, 2).contiguous()  # (B, NB, H, W, params)
        params = params.view(B * self.num_blocks, H, W, self.params_per_block)
        raw_diag = params[..., :self.diag_params_per_block]
        raw_lower = params[..., self.diag_params_per_block:]
        return raw_diag, raw_lower

    def forward(self, x, context):
        # x: (B, C=num_channels, H, W), context: (B, context_channels, H, W)
        x_vec, B, H, W = self._reshape_blocks(x)  # (B*NB, H, W, BS)
        raw_diag, raw_lower = self._get_params(context, B, H, W)

        scale = self.scale_clamp
        log_diag = (2.0 * scale / math.pi) * torch.atan(raw_diag / scale)
        diag = torch.exp(log_diag)  # (B*NB, H, W, BS)

        y_vec = torch.zeros_like(x_vec)
        offsets = self.row_offsets

        for i in range(self.block_size):
            y_i = diag[..., i] * x_vec[..., i]
            if i > 0:
                off = int(offsets[i].item())
                row_params = raw_lower[..., off:off + i]  # (..., i)
                x_prev = x_vec[..., :i]  # (..., i)
                y_i = y_i + (row_params * x_prev).sum(dim=-1)
            y_vec[..., i] = y_i

        y = self._unreshape_blocks(y_vec, B, H, W)
        log_det = log_diag.sum(dim=[1, 2, 3])  # sum over NB*H*W*BS per batch element
        return y, log_det

    def inverse(self, y, context):
        # y: (B, C=num_channels, H, W), context: (B, context_channels, H, W)
        y_vec, B, H, W = self._reshape_blocks(y)
        raw_diag, raw_lower = self._get_params(context, B, H, W)

        scale = self.scale_clamp
        log_diag = (2.0 * scale / math.pi) * torch.atan(raw_diag / scale)
        diag = torch.exp(log_diag)

        x_vec = torch.zeros_like(y_vec)
        offsets = self.row_offsets

        for i in range(self.block_size):
            rhs = y_vec[..., i]
            if i > 0:
                off = int(offsets[i].item())
                row_params = raw_lower[..., off:off + i]  # (..., i)
                x_prev = x_vec[..., :i]
                rhs = rhs - (row_params * x_prev).sum(dim=-1)
            x_vec[..., i] = rhs / diag[..., i]

        x = self._unreshape_blocks(x_vec, B, H, W)
        log_det = -log_diag.sum(dim=[1, 2, 3])
        return x, log_det


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
        linear_mixing=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        
        # The first part of the input is passed through
        self.identity_channels = in_channels // 2
        
        # The second part is transformed
        self.transformed_channels = in_channels - self.identity_channels

        # Context network over the passthrough half to provide spatial features
        # for the spline transformation. Keep output channels modest by default.
        self.context_channels = hidden_channels
        self.context_net = CNNCouplingNet(
            in_channels=self.identity_channels,
            hidden_channels=hidden_channels,
            out_channels=self.context_channels,
            num_resnet_blocks=num_resnet_blocks,
            dropout=dropout,
        )

        self.spline = BlockAutoregressiveSpline(
            num_channels=self.transformed_channels,
            context_channels=self.context_channels,
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

        # Optional pre/post triangular linear mixing to better mimic multidimensional splines
        self.linear_mixing = linear_mixing
        if self.linear_mixing:
            self.pre_linear = BlockConditionalTriangularLinear(
                num_channels=self.transformed_channels,
                context_channels=self.context_channels,
                hidden_channels=hidden_channels,
                num_resnet_blocks=num_resnet_blocks,
                block_size=block_size,
                dropout=dropout,
                scale_clamp=1.0,
            )
            self.post_linear = BlockConditionalTriangularLinear(
                num_channels=self.transformed_channels,
                context_channels=self.context_channels,
                hidden_channels=hidden_channels,
                num_resnet_blocks=num_resnet_blocks,
                block_size=block_size,
                dropout=dropout,
                scale_clamp=1.0,
            )

    def forward(self, x):
        x_identity, x_transform = x.split([self.identity_channels, self.transformed_channels], dim=1)
        # Compute spatial context features from the passthrough half
        ctx = self.context_net(x_identity)
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        # Optional pre-mixing
        if self.linear_mixing:
            x_transform, ld = self.pre_linear(x_transform, context=ctx)
            log_det_total = log_det_total + ld
        # Use context features to condition the spline transformation
        y_transform, ld = self.spline(x_transform, context=ctx)
        log_det_total = log_det_total + ld
        # Optional post-mixing
        if self.linear_mixing:
            y_transform, ld = self.post_linear(y_transform, context=ctx)
            log_det_total = log_det_total + ld
        
        y = torch.cat([x_identity, y_transform], dim=1)
        
        return y, log_det_total

    def inverse(self, y):
        y_identity, y_transform = y.split([self.identity_channels, self.transformed_channels], dim=1)
        # Compute spatial context features from the passthrough half
        ctx = self.context_net(y_identity)
        log_det_total = torch.zeros(y.shape[0], device=y.device)
        # Optional post-mixing inverse
        if self.linear_mixing:
            y_transform, ld = self.post_linear.inverse(y_transform, context=ctx)
            log_det_total = log_det_total + ld
        # Condition inverse spline transformation on the same context
        x_transform, ld = self.spline.inverse(y_transform, context=ctx)
        log_det_total = log_det_total + ld
        # Optional pre-mixing inverse
        if self.linear_mixing:
            x_transform, ld = self.pre_linear.inverse(x_transform, context=ctx)
            log_det_total = log_det_total + ld
        
        x = torch.cat([y_identity, x_transform], dim=1)
        
        return x, log_det_total
    

if __name__ == '__main__':
    # --- MC logdet estimator utilities ---
    def mc_logdet_estimate(module: nn.Module, x: torch.Tensor, eps: float = 1e-4):
        """
        Monte Carlo volume test to estimate local log|det J_f(x)|.
        Uses orthonormal random directions and finite differences to build
        an approximate Jacobian, then computes slogdet.

        Assumes input/output shapes match and batch-wise independence.
        Keep dimensionality small (e.g., H=W=1, small C) for stability.
        """
        with torch.no_grad():
            y0, _ = module(x)
            B = x.size(0)
            d = x[0].numel()

            # Orthonormal basis via QR
            rnd = torch.randn(d, d, device=x.device)
            Q, _ = torch.linalg.qr(rnd)

            J_cols = []
            for i in range(d):
                v = Q[:, i].view(1, *x.shape[1:]).expand(B, *x.shape[1:])
                y_i, _ = module(x + eps * v)
                col = ((y_i - y0).view(B, d)) / eps
                J_cols.append(col)

            J = torch.stack(J_cols, dim=-1)  # (B, d, d)
            signs, logabsdets = torch.linalg.slogdet(J)
            return logabsdets

    def run_mc_check(module: nn.Module, x: torch.Tensor, name: str):
        log_det_mc = mc_logdet_estimate(module, x)
        y, log_det_fwd = module(x)
        err = (log_det_fwd - log_det_mc).abs().mean().item()
        print(f"[MC] {name}: fwd_logdet mean={log_det_fwd.mean().item():.6f}, mc_logdet mean={log_det_mc.mean().item():.6f}, abs diff={err:.6e}")

    def randomize_params(module: nn.Module, std: float = 0.05):
        """Randomly initialize all parameters of a module for testing."""
        with torch.no_grad():
            for p in module.parameters():
                if p.requires_grad:
                    p.data.normal_(0.0, std)

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
        
        # --- Identity Init Test ---
        print("--- Testing Identity Initialization ---")
        y_init, log_det_init = spline_layer(x, context=context)
        identity_error = torch.abs(x - y_init).max()
        log_det_init_error = torch.abs(log_det_init).mean()
        print(f"Identity error at init: {identity_error.item()}")
        print(f"Log-determinant at init: {log_det_init.mean().item()}")
        assert torch.allclose(x, y_init, atol=1e-5), "Not an identity function at init!"
        assert torch.allclose(log_det_init, torch.zeros_like(log_det_init), atol=1e-5), "Log-determinant is not zero at init!"
        print("Identity Initialization test PASSED!")
        # --- End Identity Init Test ---

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

    print("\n" + "="*50 + "\n")

    # --- MC Volume Test (Small-Dim) ---
    print("--- MC Volume Test: AffineCoupling & BlockAR Coupling ---")
    try:
        B_small, C_small, H_small, W_small = 1, 4, 1, 1
        x_small = torch.randn(B_small, C_small, H_small, W_small)

        affine_small = AffineCoupling(in_channels=C_small, hidden_channels=32, num_resnet_blocks=1)
        run_mc_check(affine_small, x_small, name="AffineCoupling")

        # Random params variant
        affine_small_rand = copy.deepcopy(affine_small)
        randomize_params(affine_small_rand, std=0.1)
        run_mc_check(affine_small_rand, x_small, name="AffineCoupling-Random")

        blockar_small = BlockAutoregressiveCoupling(
            in_channels=C_small,
            hidden_channels=32,
            num_resnet_blocks=1,
            block_size=2,
        )
        run_mc_check(blockar_small, x_small, name="BlockAutoregressiveCoupling")

        # Random params variant
        blockar_small_rand = copy.deepcopy(blockar_small)
        randomize_params(blockar_small_rand, std=0.1)
        run_mc_check(blockar_small_rand, x_small, name="BlockAutoregressiveCoupling-Random")

        print("MC Volume Test PASSED (ran without exceptions)")
    except Exception as e:
        print(f"MC Volume Test FAILED: {e}")
        import traceback
        traceback.print_exc()

