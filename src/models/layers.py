import torch
import torch.nn as nn
import torch.linalg as la

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
        
        # Log-det calculation
        # d/dx logit(s) = (1-2a) / (s(1-s))
        # log_det = log(1-2a) - log(s) - log(1-s)
        log_det = torch.sum(
            torch.log(torch.tensor(1 - 2 * self.alpha, device=x.device))
            - torch.log(s) - torch.log(1 - s),
            dim=[1, 2, 3]
        )
        return y, log_det

    def inverse(self, y):
        # y in (-inf, inf)
        s = torch.sigmoid(y)
        x = (s - self.alpha) / (1 - 2 * self.alpha)
        
        # Log-det is negative of forward
        log_det = -torch.sum(
            torch.log(torch.tensor(1 - 2 * self.alpha, device=y.device))
            - torch.log(s) - torch.log(1 - s),
            dim=[1, 2, 3]
        )
        return x, log_det

class ActNorm(nn.Module):
    def __init__(self, num_channels, initialization="identity"):
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
                # Initialize logs = log(1/std) = -log(std)
                self.logs.data.copy_(-torch.log(std + 1e-6))
                self.bias.data.copy_(-mean)
                self.initialized.fill_(1)

        # y = x * exp(logs) + bias
        y = torch.exp(self.logs) * x + self.bias
        _, _, h, w = x.size()
        # log_det = sum(logs) * h * w
        log_det = torch.sum(self.logs) * h * w
        return y, log_det

    def inverse(self, y):
        # x = (y - bias) * exp(-logs)
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
        # Return a zero tensor with shape (batch_size,) on the correct device
        log_det = torch.zeros(b, device=x.device)
        return x, log_det
    
    def inverse(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
        # Return a zero tensor with shape (batch_size,) on the correct device
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
    x -> Conv(1x1) -> GatedAct -> Conv(3x3) -> GatedAct -> Conv(1x1) -> + x
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, kernel_size=1, padding=0)
        self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = GatedConv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return residual + out

class FlowPlusPlusCouplingNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks=2):
        super().__init__()
        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Stack of Residual Blocks
        self.blocks = nn.ModuleList([
            GatedResNetBlock(hidden_channels) 
            for _ in range(num_blocks)
        ])
        
        # Final projection to output parameters (s, t)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        
        # Zero initialization for the last layer (Identity Init)
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.out_conv(x)

class CNNCouplingLayer(nn.Module):
    scale_clamp: torch.Tensor
    
    def __init__(self, in_channels, hidden_channels=512, num_blocks=2, scale_clamp=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        
        self.coupling_net = FlowPlusPlusCouplingNet(
            self.split_size, 
            hidden_channels, 
            self.in_channels,
            num_blocks=num_blocks
        )
        
        self.register_buffer("scale_clamp", torch.tensor(scale_clamp))


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

        # LU Decomposition using PyTorch
        # torch.linalg.lu returns P, L, U
        # Note: P is returned as a permutation matrix in recent versions, 
        # or pivots in older ones. Let's assume recent torch.
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
        w_inv = torch.inverse(w.squeeze()).view(w.size(0), w.size(1), 1, 1)
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
    pass