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
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
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
                self.scale.data.copy_(1.0 / (std + 1e-6))
                self.bias.data.copy_(-mean)
                self.initialized.fill_(1)

        y = self.scale * x + self.bias
        _, _, h, w = x.size()
        log_det = torch.sum(torch.log(torch.abs(self.scale))) * h * w
        return y, log_det

    def inverse(self, y):
        x = (y - self.bias) / self.scale
        _, _, h, w = y.size()
        log_det = -torch.sum(torch.log(torch.abs(self.scale))) * h * w
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
    
class BottleneckResNetBlock(nn.Module):
    """A bottleneck residual block for the coupling network."""
    def __init__(self, channels, bottleneck_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(x))
        out = self.conv2(self.relu(out))
        out = self.conv3(out)
        out += residual
        return out
    
class GatedConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2 * hidden_channels, 3, padding=1),
            nn.GLU(dim=1),
            
            nn.Conv2d(hidden_channels, 2 * hidden_channels, 1),
            nn.GLU(dim=1),
            
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
        
        # Zero initialization for the last layer (Identity Init)
        self.net[-1].weight.data.zero_() # type: ignore
        self.net[-1].bias.data.zero_() # type: ignore

    def forward(self, x):
        return self.net(x)

class CNNCouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        
        self.coupling_net = GatedConvNet(self.split_size, hidden_channels, self.in_channels)
                    
        self.scale_clamp = nn.Parameter(torch.tensor(1.0)) 


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
        log_det = -torch.sum(torch.log(s), dim=[1, 2, 3])
        return x, log_det

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels, initialization="orthogonal"):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        
        if initialization == "orthogonal":
            W = la.qr(torch.randn(num_channels, num_channels))[0]
        elif initialization == "identity":
            W = torch.eye(num_channels)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        self.conv.weight.data.copy_(W.view(num_channels, num_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(1, dtype=torch.uint8))

    def forward(self, x):
        y = self.conv(x)
        _, _, h, w = x.size()
        log_det = torch.slogdet(self.conv.weight.squeeze())[1] * h * w
        return y, log_det

    def inverse(self, y):
        W_inv = torch.inverse(self.conv.weight.squeeze()).view(self.conv.in_channels, self.conv.in_channels, 1, 1)
        x = nn.functional.conv2d(y, W_inv)
        _, _, h, w = y.size()
        log_det = -torch.slogdet(self.conv.weight.squeeze())[1] * h * w
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