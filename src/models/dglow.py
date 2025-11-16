import torch
import torch.nn as nn
from .layers import ActNorm, Invertible1x1Conv, CNNCouplingLayer, Squeeze, Split

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, bottleneck_channels, num_res_blocks):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.inv_conv = Invertible1x1Conv(in_channels)
        self.coupling = CNNCouplingLayer(
            in_channels, 
            hidden_channels, 
            bottleneck_channels=bottleneck_channels, 
            num_res_blocks=num_res_blocks
        )

    def forward(self, x):
        x, log_det_act = self.actnorm(x)
        x, log_det_conv = self.inv_conv(x)
        x, log_det_coup = self.coupling(x)
        return x, log_det_act + log_det_conv + log_det_coup
    
    def inverse(self, z):
        z, log_det_coup = self.coupling.inverse(z)
        z, log_det_conv = self.inv_conv.inverse(z)
        z, log_det_act = self.actnorm.inverse(z)
        return z, log_det_act + log_det_conv + log_det_coup

class DGLOWNetwork(nn.Module):
    def __init__(self, in_channels: int, num_levels: int, steps_per_level: int, hidden_channels: int, bottleneck_channels: int, num_res_blocks: int):
        super(DGLOWNetwork, self).__init__()
        self.squeeze = Squeeze()
        self.levels = nn.ModuleList()
        self.num_levels = num_levels
        self.steps_per_level = steps_per_level
        
        current_channels = in_channels
        for _ in range(num_levels):
            current_channels *= 4  # After Squeeze
            level_flows = nn.ModuleList([
                FlowStep(
                    current_channels, 
                    hidden_channels, 
                    bottleneck_channels=bottleneck_channels, 
                    num_res_blocks=num_res_blocks
                ) for _ in range(steps_per_level)
            ])
            self.levels.append(level_flows)
            
            split = Split(current_channels)
            self.levels.append(split)
            current_channels //= 2 # After Split
            
        self.levels = self.levels[:-1]  # Remove last split

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)

        total_log_det = torch.zeros(x.shape[0], device=x.device)
        outputs = []
        z_out = []
        
        for level in self.levels:
            if isinstance(level, nn.ModuleList): # Flow steps
                x, log_det_squeeze = self.squeeze(x)
                total_log_det += log_det_squeeze
                
                for flow_step in level:
                    x, log_det = flow_step(x)
                    total_log_det += log_det
                    
                    x_out = x.flatten(start_dim=1)
                    x_out = torch.cat([x_out] + z_out, dim=1)
                    outputs.append((x_out, total_log_det.clone()))
            elif isinstance(level, Split): # Split
                x, z = level(x)
                z_out.append(z.flatten(start_dim=1))

        return outputs
    
    def inverse(self, z_final):
        total_log_det = torch.zeros(z_final.shape[0], device=z_final.device)
        
        z_shapes = []
        c, h, w = 1, 28, 28
        temp_c = c
        for i in range(self.num_levels):
            temp_c, h, w = temp_c * 4, h // 2, w // 2
            if i < self.num_levels - 1:
                z_shapes.append((temp_c // 2, h, w))
                temp_c = temp_c // 2
            else:
                z_shapes.insert(0, (temp_c, h, w))
        
        split_sizes = [s[0]*s[1]*s[2] for s in z_shapes]
        
        z_parts = z_final.split(split_sizes, dim=1)
        
        x = z_parts[0].view(z_final.shape[0], *z_shapes[0])
        z_stack = [p.view(z_final.shape[0], *s) for p, s in zip(z_parts[1:], z_shapes[1:])]

        for level in reversed(self.levels):
            if isinstance(level, nn.ModuleList): # Flow steps
                for flow_step in reversed(level):
                    x, log_det = flow_step.inverse(x)
                    total_log_det += log_det
                    
                x, log_det_squeeze = self.squeeze.inverse(x)
                total_log_det += log_det_squeeze
            elif isinstance(level, Split): # Split
                z = z_stack.pop()
                x, log_det = level.inverse(x, z)
                total_log_det += log_det
                
        return x, total_log_det
    
    @property
    def total_supervision_layers(self):
        return self.num_levels * self.steps_per_level

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Test FlowStep Invertibility ---
    print("\n--- Testing FlowStep Invertibility ---")
    
    # Define parameters for the test
    test_in_channels = 8
    test_hidden_channels = 32
    test_bottleneck_channels = 16
    test_num_res_blocks = 1
    
    flow_step_test = FlowStep(
        in_channels=test_in_channels,
        hidden_channels=test_hidden_channels,
        bottleneck_channels=test_bottleneck_channels,
        num_res_blocks=test_num_res_blocks
    ).to(device)
    
    x_test = torch.randn(4, test_in_channels, 16, 16).to(device)
    
    try:
        # Forward pass
        y_test, log_det_fwd = flow_step_test.forward(x_test)
        
        # Inverse pass
        x_recon_test, log_det_inv = flow_step_test.inverse(y_test)
        
        # Check reconstruction error
        recon_error = torch.abs(x_test - x_recon_test).mean().item()
        print(f"  Reconstruction Error: {recon_error:.2e}")

        # Check log-determinant
        log_det_error = torch.abs(log_det_fwd + log_det_inv).mean().item()
        print(f"  Log-Determinant Sum: {log_det_error:.2e}")

        if recon_error > 1e-5 or log_det_error > 1e-5:
            print("  [STATUS] FlowStep FAILED")
        else:
            print("  [STATUS] FlowStep PASSED")
    except Exception as e:
        print(f"  [STATUS] FlowStep FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        
    # --- Test DGLOWNetwork Invertibility (1 Level) ---
    num_levels = 2
    print(f"\n--- Testing DGLOWNetwork Invertibility ({num_levels} Level) ---")
    try:
        model = DGLOWNetwork(
            in_channels=1,
            num_levels=num_levels,
            steps_per_level=6,
            hidden_channels=32,
            bottleneck_channels=16,
            num_res_blocks=1
        ).to(device)
        
        x_test = torch.randn(4, 1, 28, 28).to(device)
        
        # Forward pass
        outputs = model.forward(x_test)
        z_final, log_det_fwd = outputs[-1]
        
        # Inverse pass
        x_recon, log_det_inv = model.inverse(z_final)
        
        # Check reconstruction error
        recon_error = torch.abs(x_test - x_recon).mean().item()
        print(f"  Reconstruction Error: {recon_error:.2e}")

        # Check log-determinant
        log_det_error = torch.abs(log_det_fwd + log_det_inv).mean().item()
        print(f"  Log-Determinant Sum: {log_det_error:.2e}")

        if recon_error > 1e-5 or log_det_error > 1e-5:
            print(f"  [STATUS] DGLOWNetwork ({num_levels} Level) FAILED")
        else:
            print(f"  [STATUS] DGLOWNetwork ({num_levels} Level) PASSED")
    except Exception as e:
        print(f"  [STATUS] DGLOWNetwork ({num_levels} Level) FAILED with exception: {e}")
        import traceback
        traceback.print_exc()