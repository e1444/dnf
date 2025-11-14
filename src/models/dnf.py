import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as la
from torchvision import datasets, transforms

class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

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

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return x, 0
    
class BottleneckResNetBlock(nn.Module):
    """A bottleneck residual block for the coupling network."""
    def __init__(self, channels, bottleneck_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out += residual
        out = self.relu(out)
        return out

class CNNCouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, bottleneck_channels=128, num_res_blocks=3):
        super().__init__()
        self.in_channels = in_channels
        self.split_size = in_channels // 2

        self.coupling_net = nn.Sequential(
            nn.Conv2d(self.split_size, hidden_channels, kernel_size=3, padding=1),
            *[BottleneckResNetBlock(hidden_channels, bottleneck_channels) for _ in range(num_res_blocks)],
            nn.Conv2d(hidden_channels, self.in_channels, kernel_size=3, padding=1)
        )
        self.coupling_net[-1].weight.data.zero_()
        self.coupling_net[-1].bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.split(self.split_size, dim=1)
        s_and_t = self.coupling_net(x_a)
        log_s, t = s_and_t.split(self.split_size, dim=1)
        s = torch.exp(torch.tanh(log_s))

        y_b = x_b * s + t
        y = torch.cat([x_a, y_b], dim=1)
        log_det = torch.sum(torch.log(s), dim=[1, 2, 3])
        return y, log_det

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        W = la.qr(torch.randn(num_channels, num_channels))[0]
        self.conv.weight.data.copy_(W.view(num_channels, num_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(1, dtype=torch.uint8))

    def forward(self, x):
        y = self.conv(x)
        _, _, h, w = x.size()
        log_det = torch.slogdet(self.conv.weight.squeeze())[1] * h * w
        return y, log_det

class DNFNetwork(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int):
        super(DNFNetwork, self).__init__()
        self.squeeze = Squeeze()
        current_channels = in_channels * 4
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ActNorm(current_channels))
            self.layers.append(Invertible1x1Conv(current_channels))
            self.layers.append(CNNCouplingLayer(current_channels, hidden_channels=hidden_channels))

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)

        total_log_det = torch.zeros(x.shape[0], device=x.device)
        x, log_det = self.squeeze(x)
        total_log_det += log_det
        intermediate_outputs = []

        for layer in self.layers:
            x, log_det = layer(x)
            total_log_det += log_det
            if isinstance(layer, CNNCouplingLayer):
                intermediate_outputs.append((x.flatten(start_dim=1), total_log_det.clone()))

        return intermediate_outputs