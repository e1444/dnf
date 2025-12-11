import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

from src.distributions.kpmvn import KroneckerProductMVN
from src.models.modules import AttentionPooling
from src.utils.flat_dist import FlattenedDistribution


class LowRankMVNPrior(nn.Module):
    def __init__(self, loc: torch.Tensor, diag: torch.Tensor, U: torch.Tensor):
        """
        Low-Rank Multivariate Normal Prior

        Args:
            loc (torch.Tensor): Mean vector of shape (D,)
            diag (torch.Tensor): Diagonal covariance of shape (D,)
            U (torch.Tensor): Low-rank component of shape (D, r)
        """
        super(LowRankMVNPrior, self).__init__()
        
        assert (diag > 0).all(), "Diagonal covariance must be positive"
        
        self.loc = nn.Parameter(loc)
        self.log_diag = nn.Parameter(torch.log(diag))
        self.U = nn.Parameter(U)

    def forward(self, cov_scale: float = 1.0, unit_scale: bool = False, eps: float = 1e-6):
        if unit_scale:
            if cov_scale != 1.0:
                raise Warning("cov_scale is ignored when unit_scale=True")
            
            dim = self.loc.shape[0]
            cov_scale = torch.exp(-self.log_det / dim)  # type: ignore
            
        base_dist = LowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.U * (cov_scale ** 0.5),
            cov_diag=torch.exp(self.log_diag) * cov_scale + eps
        )
        return FlattenedDistribution(base_dist)
        
    @staticmethod
    def _compute_log_det(cov_diag, cov_factor):
        """
        Computes log determinant of a low-rank matrix D + UU^T via the matrix determinant lemma.
        """
        r = cov_factor.shape[1]
        D_inv = 1.0 / cov_diag
        # M = I_r + U^T D^{-1} U
        U_scaled = cov_factor * D_inv.unsqueeze(-1)
        M = torch.eye(r, device=cov_diag.device, dtype=cov_diag.dtype) + cov_factor.t() @ U_scaled
        sign, logabsdet_M = torch.linalg.slogdet(M)
        logdet_D = torch.sum(torch.log(cov_diag))
        return logdet_D + logabsdet_M

    @property
    def log_det(self, eps: float = 1e-6):
        D = torch.exp(self.log_diag) + eps
        U = self.U
        log_det = LowRankMVNPrior._compute_log_det(D, U)
        return log_det
        

class KPMVNPrior(nn.Module):
    def __init__(
        self,
        loc: torch.Tensor,
        ch_cov: tuple,
        sp_cov: tuple,
        C: int,
        H: int,
        W: int,
        jitter: float = 1e-6
    ):
        """
        Kronecker-Product Multivariate Normal Prior
        
        Args:
            loc: (C * S,) mean vector
            ch_cov: tuple (U_ch, D_ch) where U_ch: (C, r_ch), D_ch: (C,)
            sp_cov: tuple (U_sp, D_sp) where U_sp: (S, r_sp), D_sp: (S,)
        """
        super(KPMVNPrior, self).__init__()
        self.C, self.H, self.W = C, H, W
        self.D_ch = C
        self.D_sp = H * W
        
        assert (ch_cov[1] > 0).all(), "Channel diagonal must be positive"
        assert (sp_cov[1] > 0).all(), "Spatial diagonal must be positive"

        self._loc = nn.Parameter(loc)
        self.ch_cov_factor = nn.Parameter(ch_cov[0])
        self.log_ch_cov_diag = nn.Parameter(torch.log(ch_cov[1]))
        self.sp_cov_factor = nn.Parameter(sp_cov[0])
        self.log_sp_cov_diag = nn.Parameter(torch.log(sp_cov[1]))

        self.jitter = jitter
        
    def forward(self, ch_cov_scale: float = 1.0, sp_cov_scale: float = 1.0, unit_scale: bool = False, eps: float = 1e-6):
        if unit_scale:
            if ch_cov_scale != 1.0:
                raise Warning("ch_cov_scale is ignored when unit_scale=True")
            if sp_cov_scale != 1.0:
                raise Warning("sp_cov_scale is ignored when unit_scale=True")
            
            ch_cov_scale = torch.exp(-self.log_det_ch / self.D_ch)  # type: ignore
            sp_cov_scale = torch.exp(-self.log_det_sp / self.D_sp)  # type: ignore
            
        return KroneckerProductMVN(
            loc=self._loc,
            ch_cov=(self.ch_cov_factor * (ch_cov_scale ** 0.5), torch.exp(self.log_ch_cov_diag) * ch_cov_scale + eps),
            sp_cov=(self.sp_cov_factor * (sp_cov_scale ** 0.5), torch.exp(self.log_sp_cov_diag) * sp_cov_scale + eps),
            C=self.C,
            H=self.H,
            W=self.W,
            jitter=self.jitter
        )
        
    @property
    def log_det_ch(self, eps: float = 1e-6):
        return KroneckerProductMVN._compute_log_det(torch.exp(self.log_ch_cov_diag) + eps, self.ch_cov_factor)

    @property
    def log_det_sp(self, eps: float = 1e-6):
        return KroneckerProductMVN._compute_log_det(torch.exp(self.log_sp_cov_diag) + eps, self.sp_cov_factor)

    @property
    def log_det(self, eps: float = 1e-6):
        log_det_ch = KroneckerProductMVN._compute_log_det(torch.exp(self.log_ch_cov_diag) + eps, self.ch_cov_factor)
        log_det_sp = KroneckerProductMVN._compute_log_det(torch.exp(self.log_sp_cov_diag) + eps, self.sp_cov_factor)
        log_det_total = self.D_sp * log_det_ch + self.D_ch * log_det_sp
        return log_det_total
        
        
class ClassConditionalPrior(nn.Module):
    def __init__(self, priors: list[nn.Module]):
        """
        Class-Conditional Prior

        Args:
            priors (list[nn.Module]): List of prior distributions for each class
        """
        super(ClassConditionalPrior, self).__init__()
        self.priors = nn.ModuleList(priors)

    def forward(self, unit_scale: bool = False):
        if not unit_scale:
            return [prior() for prior in self.priors]

        distributions = []
        for prior in self.priors:
            if isinstance(prior, KPMVNPrior):
                # Separately scale channel and spatial components to have unit determinant
                # log_det(s*Σ) = D*log(s) + log_det(Σ) = 0  =>  s = exp(-log_det(Σ)/D)
                ch_scale = torch.exp(-prior.log_det_ch / prior.D_ch)
                sp_scale = torch.exp(-prior.log_det_sp / prior.D_sp)
                dist = prior(ch_cov_scale=ch_scale, sp_cov_scale=sp_scale)
            elif isinstance(prior, LowRankMVNPrior):
                # Scale the entire covariance to have unit determinant
                dim = prior.loc.shape[0]
                scale = torch.exp(-prior.log_det / dim)
                dist = prior(cov_scale=scale)
            else:
                # Default behavior for other prior types
                dist = prior(unit_scale=True)
            
            distributions.append(dist)
            
        return distributions


class ConditionalKPMVNPrior(nn.Module):
    def __init__(
        self,
        z_channels: int,
        h_channels: int,
        H: int, W: int,
        rank_ch: int, rank_sp: int,
        backbone_features: int = 256,
    ):
        """
        A conditional prior p(h | z) that predicts the parameters for the
        distribution of h, conditioned on z.

        Args:
            z_channels: Number of channels in the conditioning tensor z.
            h_channels: Number of channels in the target tensor h.
            H, W: Spatial dimensions of the latent variables.
            rank_ch, rank_sp: Ranks for the KPMVN covariance factors.
            backbone_features: Number of features in the backbone network.
        """
        super().__init__()
        self.h_C, self.H, self.W = h_channels, H, W
        self.S = H * W
        self.rank_ch, self.rank_sp = rank_ch, rank_sp

        # 1. Shared Backbone to process z
        self.backbone = nn.Sequential(
            nn.Conv2d(z_channels, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
        )

        # 2. Heads for Spatial Parameters of h
        self.loc_head = nn.Conv2d(backbone_features, self.h_C, 1)
        self.sp_D_head = nn.Conv2d(backbone_features, 1, 1)
        self.sp_U_head = nn.Conv2d(backbone_features, self.rank_sp, 1)

        # 3. Attention Pooling and Heads for Non-Spatial Parameters of h
        self.attention_pool = AttentionPooling(backbone_features)
        self.ch_D_head = nn.Linear(backbone_features, self.h_C)
        self.ch_U_head = nn.Linear(backbone_features, self.h_C * self.rank_ch)

    def forward(self, z: torch.Tensor, ch_cov_scale: float = 1.0, sp_cov_scale: float = 1.0, unit_scale: bool = False, eps: float = 1e-6) -> KroneckerProductMVN:
        """
        Takes a batch of conditioning tensors (z) and returns a batch of
        distributions for h.

        Args:
            z: The conditioning tensor, shape (B, z_channels, H, W).

        Returns:
            A KroneckerProductMVN distribution object for h, with batch size B.
        """
        B = z.shape[0]
        shared_features = self.backbone(z)

        # --- Predict Spatial Parameters for h ---
        loc = self.loc_head(shared_features).view(B, -1) # (B, h_C * S)
        log_sp_D = self.sp_D_head(shared_features).view(B, -1) # (B, S)
        sp_U = self.sp_U_head(shared_features).permute(0, 2, 3, 1).reshape(B, self.S, self.rank_sp)

        # --- Predict Non-Spatial Parameters for h ---
        pooled_features = self.attention_pool(shared_features)
        log_ch_D = self.ch_D_head(pooled_features)
        ch_U = self.ch_U_head(pooled_features).view(B, self.h_C, self.rank_ch)
        
        if unit_scale:
            if ch_cov_scale != 1.0:
                raise Warning("ch_cov_scale is ignored when unit_scale=True")
            if sp_cov_scale != 1.0:
                raise Warning("sp_cov_scale is ignored when unit_scale=True")
            
            # Scale channel and spatial covariances to have unit determinant
            ch_cov_scale = torch.exp(-KroneckerProductMVN._compute_log_det(torch.exp(log_ch_D), ch_U) / self.h_C)   # type: ignore
            sp_cov_scale = torch.exp(-KroneckerProductMVN._compute_log_det(torch.exp(log_sp_D), sp_U) / self.S)     # type: ignore

        # Instantiate and return the batched distribution for h
        return KroneckerProductMVN(
            loc=loc,
            ch_cov=(ch_U * (ch_cov_scale ** 0.5), torch.exp(log_ch_D) * ch_cov_scale + eps),
            sp_cov=(sp_U * (sp_cov_scale ** 0.5), torch.exp(log_sp_D) * sp_cov_scale + eps),
            C=self.h_C, H=self.H, W=self.W
        )