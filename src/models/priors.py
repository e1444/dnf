import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

from src.distributions.kpmvn import KroneckerProductMVN
from src.models.modules import AttentionPooling, GaussianBlurLayer
from src.utils.flat_dist import FlattenedDistribution

from typing import List


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
        cov_scale = torch.exp((self.tau - self._log_det) / dim)
            
        base_dist = LowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.U * (cov_scale ** 0.5),
            cov_diag=torch.exp(self.log_diag) * cov_scale + self.eps
        )
        return FlattenedDistribution(base_dist)
        
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
        tau: torch.Tensor, # Single target!
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
        
        # Single learnable target for the total entropy
        self.tau = nn.Parameter(tau)
        
        self.jitter = jitter
        self.eps = eps
        
    def forward(self) -> KroneckerProductMVN:
        # Calculate current total log det
        current_log_det = self._log_det
        
        log_s = (self.tau - current_log_det) / self.D_total
        s = torch.exp(log_s)
        scale_factor = s ** 0.5
            
        return KroneckerProductMVN(
            loc=self._loc,
            ch_cov=(self.cov_ch_factor * (scale_factor ** 0.5), torch.exp(self.log_cov_ch_diag) * scale_factor + self.eps),
            sp_cov=(self.cov_sp_factor * (scale_factor ** 0.5), torch.exp(self.log_cov_sp_diag) * scale_factor + self.eps),
            C=self.C,
            H=self.H,
            W=self.W,
            jitter=self.jitter
        )
        
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
        tau: torch.Tensor,
        jitter: float = 1e-6,
        eps: float = 1e-6,
        dropout: float = 0.0,
        blur_sigma: float = 0.0,
        backbone_features: int = 256,
    ):
        super().__init__()
        self.h_C, self.H, self.W = h_channels, H, W
        self.S = H * W
        self.D_total = self.h_C * self.S
        self.rank_ch, self.rank_sp = rank
        self.jitter = jitter
        self.eps = eps
        
        self.tau = nn.Parameter(tau)

        self.blur = nn.Identity()
        if blur_sigma > 0.0:
            self.blur = GaussianBlurLayer(channels=z_channels, kernel_size=5, sigma=blur_sigma)

        self.backbone = nn.Sequential(
            nn.Conv2d(z_channels, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.attention_pool = AttentionPooling(backbone_features)

        self.loc_head = nn.Conv2d(backbone_features * 2, self.h_C, 1)
        self.sp_D_head = nn.Conv2d(backbone_features * 2, 1, 1)
        self.sp_U_head = nn.Conv2d(backbone_features * 2, self.rank_sp, 1)

        self.ch_D_head = nn.Linear(backbone_features, self.h_C)
        self.ch_U_head = nn.Linear(backbone_features, self.h_C * self.rank_ch)

    def forward(self, z: torch.Tensor) -> KroneckerProductMVN:
        B = z.shape[0]
        z = self.blur(z)
        
        shared_features = self.backbone(z)
        pooled_features = self.attention_pool(shared_features)

        log_ch_D = self.ch_D_head(pooled_features)
        ch_U = self.ch_U_head(pooled_features).view(B, self.h_C, self.rank_ch)

        global_context_map = pooled_features.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        spatial_input = torch.cat([shared_features, global_context_map], dim=1)
        
        loc = self.loc_head(spatial_input).view(B, -1)
        log_sp_D = self.sp_D_head(spatial_input).view(B, -1)
        sp_U = self.sp_U_head(spatial_input).permute(0, 2, 3, 1).reshape(B, self.S, self.rank_sp)
        
        # Compute log determinants (B,)
        ld_ch = KroneckerProductMVN._compute_log_det(torch.exp(log_ch_D) + self.eps, ch_U)
        ld_sp = KroneckerProductMVN._compute_log_det(torch.exp(log_sp_D) + self.eps, sp_U)
        
        current_log_det = self.S * ld_ch + self.h_C * ld_sp
        
        # Calculate global scale s (B,)
        log_s = (self.tau - current_log_det) / self.D_total
        s = torch.exp(log_s)
        scale_factor = s ** 0.5
        
        # Reshape for broadcasting
        scale_D = scale_factor.view(-1, 1)
        scale_U = scale_factor.view(-1, 1, 1)

        return KroneckerProductMVN(
            loc=loc,
            ch_cov=(ch_U * (scale_U ** 0.5), torch.exp(log_ch_D) * scale_D + self.eps),
            sp_cov=(sp_U * (scale_U ** 0.5), torch.exp(log_sp_D) * scale_D + self.eps),
            C=self.h_C, H=self.H, W=self.W,
            jitter=self.jitter
        )