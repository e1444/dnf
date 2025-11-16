import torch
import torch.nn as nn
import torch.linalg as la

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

    def inverse(self, y):
        y_a, y_b = y.split(self.split_size, dim=1)
        s_and_t = self.coupling_net(y_a)
        log_s, t = s_and_t.split(self.split_size, dim=1)
        s = torch.exp(torch.tanh(log_s))
        
        x_b = (y_b - t) / s
        x = torch.cat([y_a, x_b], dim=1)
        log_det = -torch.sum(torch.log(s), dim=[1, 2, 3])
        return x, log_det

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
    print("--- Testing Invertible Layers ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def test_layer(layer, x, layer_name):
        print(f"\n[TESTING] {layer_name}")
        try:
            layer = layer.to(device)
            x = x.to(device)

            # Forward pass
            y, log_det_fwd = layer.forward(x)
            
            # Inverse pass
            x_recon, log_det_inv = layer.inverse(y)

            # Check reconstruction error
            recon_error = torch.abs(x - x_recon).mean().item()
            print(f"  Reconstruction Error: {recon_error:.2e}")

            # Check log-determinant
            log_det_error = torch.abs(log_det_fwd + log_det_inv).mean().item()
            print(f"  Log-Determinant Sum: {log_det_error:.2e}")

            if recon_error > 1e-6 or log_det_error > 1e-6:
                print("  [STATUS] FAILED")
                return False
            else:
                print("  [STATUS] PASSED")
                return True
        except Exception as e:
            print(f"  [STATUS] FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --- Test Cases ---
    B, H, W = 4, 16, 16
    all_passed = True

    # Test ActNorm
    C_an = 8
    all_passed &= test_layer(ActNorm(C_an), torch.randn(B, C_an, H, W), "ActNorm")

    # Test Squeeze
    C_sq = 2
    all_passed &= test_layer(Squeeze(), torch.randn(B, C_sq, H, W), "Squeeze")

    # Test Invertible1x1Conv
    C_1x1 = 12
    all_passed &= test_layer(Invertible1x1Conv(C_1x1), torch.randn(B, C_1x1, H, W), "Invertible1x1Conv")

    # Test CNNCouplingLayer
    C_cpl = 16 # Must be even
    all_passed &= test_layer(CNNCouplingLayer(C_cpl), torch.randn(B, C_cpl, H, W), "CNNCouplingLayer")

    # Special test for Split layer
    print("\n[TESTING] Split")
    try:
        C_spl = 20 # Must be even
        split_layer = Split(C_spl).to(device)
        x_split = torch.randn(B, C_spl, H, W).to(device)
        x1, x2 = split_layer.forward(x_split)
        x_recon_split, _ = split_layer.inverse(x1, x2)
        
        recon_error_split = torch.abs(x_split - x_recon_split).mean().item()
        print(f"  Reconstruction Error: {recon_error_split:.2e}")
        
        if recon_error_split > 1e-6:
            print("  [STATUS] FAILED")
            all_passed = False
        else:
            print("  [STATUS] PASSED")
    except Exception as e:
        print(f"  [STATUS] FAILED with exception: {e}")
        all_passed = False


    print("\n--- SUMMARY ---")
    if all_passed:
        print("All layers passed the invertibility test!")
    else:
        print("One or more layers FAILED the invertibility test.")