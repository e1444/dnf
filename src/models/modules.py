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


class ActNorm2d(nn.Module):
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


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
        do_actnorm=True,
        weight_std=0.0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(not do_actnorm),
        )
        
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm


    def forward(self, x):
        x = self.conv(x)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
        logscale_factor=3,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class AffineCouplingLayer(nn.Module):
    scale_clamp: torch.Tensor
    
    def __init__(self, in_channels, hidden_channels=512, scale_clamp=1.0):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be even"
        self.in_channels = in_channels
        self.split_size = in_channels // 2
        
        self.actnorm = ActNorm2d(in_channels)
        self.permute = Invertible1x1Conv(in_channels)
        self.coupling_net = nn.Sequential(
            Conv2d(self.split_size, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            Conv2dZeros(hidden_channels, self.split_size * 2, kernel_size=3, padding=1)
        )
        self.split = Split()
        
        self.register_buffer("scale_clamp", torch.tensor(scale_clamp))

    def forward(self, x):
        total_log_det = torch.zeros(x.size(0), device=x.device)
        x, log_det = self.actnorm(x)
        total_log_det = total_log_det + log_det
        x, log_det = self.permute(x)
        total_log_det = total_log_det + log_det
        
        x_a, x_b = self.split(x, method="split")
        s_and_t = self.coupling_net(x_a)
        log_s, t = self.split(s_and_t, method="cross")
        
        scale = self.scale_clamp
        log_s = (2.0 * scale / torch.pi) * torch.atan(log_s / scale)
        s = torch.exp(log_s)

        y_b = x_b * s + t
        y = torch.cat([x_a, y_b], dim=1)
        log_det = torch.sum(torch.log(s), dim=[1, 2, 3])
        total_log_det = total_log_det + log_det
        return y, total_log_det

    def inverse(self, y):
        y_a, y_b = self.split(y, method="split")
        s_and_t = self.coupling_net(y_a)
        log_s, t = self.split(s_and_t, method="cross")
        
        scale = self.scale_clamp
        log_s = (2.0 * scale / torch.pi) * torch.atan(log_s / scale)
        s = torch.exp(log_s)
        
        x_b = (y_b - t) / s
        x = torch.cat([y_a, x_b], dim=1)
        log_det = -torch.sum(log_s, dim=[1, 2, 3])
        
        x, log_det_perm = self.permute.inverse(x)
        log_det = log_det + log_det_perm
        
        x, log_det_act = self.actnorm.inverse(x)
        log_det = log_det + log_det_act

        return x, log_det


class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, method="split"):
        if method == "split":
            x1, x2 = x.chunk(2, dim=1)
            return x1, x2
        elif method == "cross":
            x1 = x[:, 0::2, ...]
            x2 = x[:, 1::2, ...]
            return x1, x2
        else:
            raise ValueError(f"Unknown split method: {method}")

    def inverse(self, x1, x2, method="split"):
        if method == "split":
            x = torch.cat((x1, x2), dim=1)
            return x, 0
        elif method == "cross":
            b, c1, h, w = x1.size()
            c2 = x2.size(1)
            c = c1 + c2
            x = torch.zeros(b, c, h, w, device=x1.device, dtype=x1.dtype)
            x[:, 0::2, ...] = x1
            x[:, 1::2, ...] = x2
            return x, 0
        else:
            raise ValueError(f"Unknown split method: {method}")


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor=2):
        b, c, h, w = x.size()
        assert h % factor == 0 and w % factor == 0, "Height and Width must be divisible by factor"
        
        x = x.view(b, c, h // factor, factor, w // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * (factor ** 2), h // factor, w // factor)
        log_det = torch.zeros(b, device=x.device)
        return x, log_det
    
    def inverse(self, x, factor=2):
        b, c, h, w = x.size()
        assert c % (factor ** 2) == 0, "Number of channels must be divisible by factor squared"
        
        x = x.view(b, c // (factor ** 2), factor, factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // (factor ** 2), h * factor, w * factor)
        log_det = torch.zeros(b, device=x.device)
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

    
if __name__ == "__main__":
    # --- Test FlowStep Invertibility ---
    print("--- Testing FlowStep Invertibility ---")
    x = torch.randn(8, 4, 4, 4).clamp(min=0.0, max=1.0)
    layer = AffineCouplingLayer(
        in_channels=4,
        hidden_channels=64,
        scale_clamp=1.0,
    )
    z, log_det = layer(x)
    x_recon, log_det_inv = layer.inverse(z)
    
    recon_error_step = torch.abs(x - x_recon).mean().item()
    log_det_error_step = torch.abs(log_det + log_det_inv).mean().item()

    print(f"FlowStep Reconstruction Error: {recon_error_step:.2e}")
    print(f"FlowStep Log-Determinant Sum: {log_det_error_step:.2e}")
    if recon_error_step < 1e-5 and log_det_error_step < 1e-5:
        print("[STATUS] PASSED")
    else:
        print("[STATUS] FAILED")