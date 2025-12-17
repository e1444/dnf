import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import LowRankMultivariateNormal

from src.distributions.kpmvn import KroneckerProductMVN
from src.models.modules import AttentionPooling, GaussianBlurLayer
from src.utils.dist_wrapper import FlattenedDistribution, ScaledDistribution

from typing import List, Union


class LowRankMVNPrior(nn.Module):
    def __init__(self, loc: torch.Tensor, diag: torch.Tensor, U: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6):
        """
        Low-Rank Multivariate Normal Prior
        """
        super(LowRankMVNPrior, self).__init__()
        
        assert (diag > 0).all(), "Diagonal covariance must be positive"
        
        self.loc = nn.Parameter(loc)
        self.log_diag = nn.Parameter(torch.log(diag))
        self.U = nn.Parameter(U)
        self.tau = nn.Parameter(tau)
        self.eps = eps

    def forward(self) -> FlattenedDistribution:
        dim = self.loc.shape[0]
        log_norm_scale = -self._log_det / dim
            
        base_dist = LowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.U * torch.exp(log_norm_scale / 2),
            cov_diag=torch.exp(self.log_diag + log_norm_scale) + self.eps
        )
        
        log_s = torch.clamp(self.tau / dim, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(base_dist, loc=self.loc, scale=global_scale)
        flattened_dist = FlattenedDistribution(scaled_dist)
    
        return flattened_dist
        
    @staticmethod
    def _compute_log_det(cov_diag, cov_factor):
        r = cov_factor.shape[1]
        D_inv = 1.0 / cov_diag
        U_scaled = cov_factor * D_inv.unsqueeze(-1)
        M = torch.eye(r, device=cov_diag.device, dtype=cov_diag.dtype) + cov_factor.t() @ U_scaled
        sign, logabsdet_M = torch.linalg.slogdet(M)
        logdet_D = torch.sum(torch.log(cov_diag))
        return logdet_D + logabsdet_M

    @property
    def _log_det(self):
        D = torch.exp(self.log_diag) + self.eps
        U = self.U
        return LowRankMVNPrior._compute_log_det(D, U)
        

class KPMVNPrior(nn.Module):
    def __init__(
        self,
        loc: torch.Tensor,
        cov_ch: tuple,
        cov_sp: tuple,
        tau: Union[float, torch.Tensor],
        C: int, H: int, W: int,
        jitter: float = 1e-6,
        eps: float = 1e-6
    ):
        """
        Kronecker-Product Multivariate Normal Prior
        """
        super(KPMVNPrior, self).__init__()
        self.C, self.H, self.W = C, H, W
        self.D_ch = C
        self.D_sp = H * W
        self.D_total = self.D_ch * self.D_sp
        
        assert (cov_ch[1] > 0).all(), "Channel diagonal must be positive"
        assert (cov_sp[1] > 0).all(), "Spatial diagonal must be positive"

        self._loc = nn.Parameter(loc)
        self.cov_ch_factor = nn.Parameter(cov_ch[0])
        self.log_cov_ch_diag = nn.Parameter(torch.log(cov_ch[1]))
        self.cov_sp_factor = nn.Parameter(cov_sp[0])
        self.log_cov_sp_diag = nn.Parameter(torch.log(cov_sp[1]))
        
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        self.tau = nn.Parameter(tau)
        
        self.jitter = jitter
        self.eps = eps
        
    def forward(self) -> torch.distributions.Distribution:
        log_norm_scale = self._log_det / self.D_total / 2
        
        loc_shaped = self._loc.view(self.C, self.H, self.W)
            
        base_dist = KroneckerProductMVN(
            loc=torch.zeros(self.C * self.H * self.W, device=self._loc.device, dtype=self._loc.dtype),
            ch_cov=(self.cov_ch_factor * torch.exp(log_norm_scale / 2), torch.exp(self.log_cov_ch_diag + log_norm_scale) + self.eps),
            sp_cov=(self.cov_sp_factor * torch.exp(log_norm_scale / 2), torch.exp(self.log_cov_sp_diag + log_norm_scale) + self.eps),
            C=self.C, H=self.H, W=self.W,
            jitter=self.jitter
        )
        
        log_s = torch.clamp(self.tau / self.D_total, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(base_dist, loc=loc_shaped, scale=global_scale)
        return scaled_dist
        
    @property
    def _log_det_ch(self):
        return KroneckerProductMVN._compute_log_det(torch.exp(self.log_cov_ch_diag) + self.eps, self.cov_ch_factor)

    @property
    def _log_det_sp(self):
        return KroneckerProductMVN._compute_log_det(torch.exp(self.log_cov_sp_diag) + self.eps, self.cov_sp_factor)

    @property
    def _log_det(self):
        return self.D_sp * self._log_det_ch + self.D_ch * self._log_det_sp
        
        
class ClassConditionalPrior(nn.Module):
    def __init__(self, priors: List[nn.Module]):
        super(ClassConditionalPrior, self).__init__()
        self.priors = nn.ModuleList(priors)
        self.prior_cls = priors[0].__class__
        self.K = len(priors)
        
        assert all(isinstance(prior, priors[0].__class__) for prior in priors), "All priors must be of the same type"

    def forward(self) -> List[torch.distributions.Distribution]:
        return [prior() for prior in self.priors]


class ConditionalKPMVNPrior(nn.Module):
    def __init__(
        self,
        z_channels: int,
        h_channels: int,
        H: int, W: int,
        rank: tuple[int, int],
        tau: Union[float, torch.Tensor],
        jitter: float = 1e-6,
        eps: float = 1e-6,
        dropout: float = 0.0,
        backbone_features: int = 256,
        cond_features: int = 0,
    ):
        super().__init__()
        self.h_C, self.H, self.W = h_channels, H, W
        self.S = H * W
        self.D_total = self.h_C * self.S
        self.rank_ch, self.rank_sp = rank
        self.jitter = jitter
        self.eps = eps
        
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        self.tau = nn.Parameter(tau)

        # Split backbone to insert FiLM
        self.conv1 = nn.Conv2d(z_channels, backbone_features, 3, padding=1)
        self.conv2 = nn.Conv2d(backbone_features, backbone_features, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.cond_features = cond_features
        assert cond_features > 0, "Conditioning feature dimension must be positive for FiLM"
        
        # Projects h -> (gamma, beta) for each feature channel
        self.film_gen = nn.Linear(cond_features, backbone_features * 2)
        # Initialize to identity: gamma=0 (scale=1), beta=0
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

        self.attention_pool = AttentionPooling(backbone_features)

        self.loc_head = nn.Conv2d(backbone_features * 2, self.h_C, 1)
        self.sp_D_head = nn.Conv2d(backbone_features * 2, 1, 1)
        self.sp_U_head = nn.Conv2d(backbone_features * 2, self.rank_sp, 1)

        self.ch_D_head = nn.Linear(backbone_features, self.h_C)
        self.ch_U_head = nn.Linear(backbone_features, self.h_C * self.rank_ch)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.distributions.Distribution:
        B = z.shape[0]
        
        # Backbone with FiLM
        x = F.relu(self.conv1(z))
        
        # h: (B, cond_features)
        film_params = self.film_gen(h)  # (B, 2 * backbone_features)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply FiLM: x = (1 + gamma) * x + beta
        x = (1 + gamma) * x + beta
            
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        shared_features = self.dropout(x)
        
        pooled_features = self.attention_pool(shared_features)

        log_ch_D = self.ch_D_head(pooled_features)
        ch_U = self.ch_U_head(pooled_features).view(B, self.h_C, self.rank_ch)

        global_context_map = pooled_features.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        spatial_input = torch.cat([shared_features, global_context_map], dim=1)
        
        loc = self.loc_head(spatial_input).view(B, -1)
        loc_shaped = loc.view(B, self.h_C, self.H, self.W)
        log_sp_D = self.sp_D_head(spatial_input).view(B, -1)
        sp_U = self.sp_U_head(spatial_input).permute(0, 2, 3, 1).reshape(B, self.S, self.rank_sp)
        
        # Compute log determinants (B,)
        ld_ch = KroneckerProductMVN._compute_log_det(torch.exp(log_ch_D) + self.eps, ch_U)
        ld_sp = KroneckerProductMVN._compute_log_det(torch.exp(log_sp_D) + self.eps, sp_U)
        
        current_log_det = self.S * ld_ch + self.h_C * ld_sp
        
        # Calculate global scale s (B,)
        log_norm_scale = torch.clamp(-current_log_det / self.D_total / 2, min=-15.0, max=15.0)
        
        # Reshape for broadcasting
        log_norm_scale_D = log_norm_scale.view(-1, 1)       # (B, 1)
        log_norm_scale_U = log_norm_scale.view(-1, 1, 1)    # (B, 1, 1)
        
        base_dist = KroneckerProductMVN(
            loc=torch.zeros(B, self.h_C * self.H * self.W, device=z.device, dtype=z.dtype),
            ch_cov=(ch_U * torch.exp(log_norm_scale_U / 2), torch.exp(log_ch_D + log_norm_scale_D) + self.eps),
            sp_cov=(sp_U * torch.exp(log_norm_scale_U / 2), torch.exp(log_sp_D + log_norm_scale_D) + self.eps),
            C=self.h_C, H=self.H, W=self.W,
            jitter=self.jitter
        )
        
        log_s = torch.clamp(self.tau / self.D_total, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(base_dist, loc=loc_shaped, scale=global_scale)
        
        return scaled_dist