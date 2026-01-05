import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

from src.distributions.kpmvn import KroneckerProductMVN
from src.distributions.kpmvt import KroneckerProductMVT
from src.models.modules import AttentionPooling
from src.utils.dist_wrapper import FlattenedDistribution, ScaledDistribution, MixtureDistribution
from src.utils.prior_init import (
    create_lrmvn_priors, 
    create_kpmvn_priors, 
    create_kpmvt_priors
)

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
        base_dist = LowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.U,
            cov_diag=self.log_diag + self.eps
        )
        
        dim = self.loc.shape[0]
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

    def __str__(self):
        return f"LowRankMVNPrior(dim={self.loc.shape[0]}, rank={self.U.shape[1]}, tau={self.tau.item():.2f})"
        

class KPMVNPrior(nn.Module):
    tau: torch.Tensor
    
    def __init__(
        self,
        C: int, H: int, W: int,
        loc: torch.Tensor,
        cov_ch: tuple,
        cov_sp: tuple,
        tau: Union[float, torch.Tensor] = 1.0,
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
        self.register_buffer("tau", tau)
        
        self.jitter = jitter
        self.eps = eps
        
    def forward(self) -> torch.distributions.Distribution:
        loc_shaped = self._loc.view(self.C, self.H, self.W)
            
        base_dist = KroneckerProductMVN(
            loc=torch.zeros(self.C * self.H * self.W, device=self._loc.device, dtype=self._loc.dtype),
            ch_cov=(self.cov_ch_factor, torch.exp(self.log_cov_ch_diag) + self.eps),
            sp_cov=(self.cov_sp_factor, torch.exp(self.log_cov_sp_diag) + self.eps),
            C=self.C, H=self.H, W=self.W,
            jitter=self.jitter
        )
        
        log_s = torch.clamp(self.tau, min=-15.0, max=15.0)
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

    def __str__(self):
        return f"KPMVNPrior(C={self.C}, H={self.H}, W={self.W}, tau={self.tau.item():.2f})"


class KPMVTPrior(nn.Module):
    tau: torch.Tensor
    
    def __init__(
        self,
        C: int, H: int, W: int,
        loc: torch.Tensor,
        cov_ch: tuple,
        cov_sp: tuple,
        tau: Union[float, torch.Tensor] = 1.0,
        df: Union[float, torch.Tensor] = 4.0,
        jitter: float = 1e-6,
        eps: float = 1e-6
    ):
        """
        Kronecker-Product Multivariate Student-T Prior
        """
        super(KPMVTPrior, self).__init__()
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
        self.register_buffer("tau", tau)

        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df, dtype=torch.float32)
        self.log_df = nn.Parameter(torch.log(df))
        
        self.jitter = jitter
        self.eps = eps
        
    def forward(self) -> torch.distributions.Distribution:
        loc_shaped = self._loc.view(self.C, self.H, self.W)
        
        # Ensure df > 2.0 for finite variance
        df = torch.exp(self.log_df) + 2.0
            
        base_dist = KroneckerProductMVT(
            loc=torch.zeros(self.C * self.H * self.W, device=self._loc.device, dtype=self._loc.dtype),
            ch_cov=(self.cov_ch_factor, torch.exp(self.log_cov_ch_diag) + self.eps),
            sp_cov=(self.cov_sp_factor, torch.exp(self.log_cov_sp_diag) + self.eps),
            df=df,
            C=self.C, H=self.H, W=self.W,
            jitter=self.jitter
        )
        
        log_s = torch.clamp(self.tau, min=-15.0, max=15.0)
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

    def __str__(self):
        df = torch.exp(self.log_df) + 2.0
        return f"KPMVTPrior(C={self.C}, H={self.H}, W={self.W}, tau={self.tau.item():.2f}, df={df.item():.2f})"


class ConditionalKPMVNPrior(nn.Module):
    tau: torch.Tensor
    
    def __init__(
        self,
        h_channels: int,
        z_channels: int,
        H: int, W: int,
        rank: tuple[int, int],
        tau: Union[float, torch.Tensor] = 1.0,
        jitter: float = 1e-6,
        eps: float = 1e-6,
        dropout: float = 0.0,
        backbone_features: int = 256,
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
        self.register_buffer("tau", tau)
        
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
        
        nn.init.zeros_(self.loc_head.weight)
        nn.init.zeros_(self.loc_head.bias)
        
        nn.init.zeros_(self.sp_D_head.weight)
        nn.init.zeros_(self.sp_D_head.bias)
        nn.init.zeros_(self.ch_D_head.weight)
        nn.init.zeros_(self.ch_D_head.bias)
        
        nn.init.normal_(self.sp_U_head.weight, std=1e-4)
        nn.init.zeros_(self.sp_U_head.bias)
        nn.init.normal_(self.ch_U_head.weight, std=1e-4)
        nn.init.zeros_(self.ch_U_head.bias)

    def forward(self, h: torch.Tensor) -> torch.distributions.Distribution:
        B = h.shape[0]
        
        shared_features = self.backbone(h)
        pooled_features = self.attention_pool(shared_features)

        log_ch_D = self.ch_D_head(pooled_features)
        ch_U = self.ch_U_head(pooled_features).view(B, self.h_C, self.rank_ch)

        global_context_map = pooled_features.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        spatial_input = torch.cat([shared_features, global_context_map], dim=1)
        
        loc = self.loc_head(spatial_input).view(B, -1)
        loc_shaped = loc.view(B, self.h_C, self.H, self.W)
        log_sp_D = self.sp_D_head(spatial_input).view(B, -1)
        sp_U = self.sp_U_head(spatial_input).permute(0, 2, 3, 1).reshape(B, self.S, self.rank_sp)
        
        base_dist = KroneckerProductMVN(
            loc=torch.zeros(B, self.h_C * self.H * self.W, device=h.device, dtype=h.dtype),
            ch_cov=(ch_U, torch.exp(log_ch_D) + self.eps),
            sp_cov=(sp_U, torch.exp(log_sp_D) + self.eps),
            C=self.h_C, H=self.H, W=self.W,
            jitter=self.jitter
        )
        
        log_s = torch.clamp(self.tau, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(base_dist, loc=loc_shaped, scale=global_scale)
        
        return scaled_dist
    
    def __str__(self):
        return f"ConditionalKPMVNPrior(h_C={self.h_C}, H={self.H}, W={self.W}, rank_ch={self.rank_ch}, rank_sp={self.rank_sp}, tau={self.tau.item():.2f})"
    
             
class ConditionalKPMVTPrior(nn.Module):
    tau: torch.Tensor
        
    def __init__(
        self,
        h_channels: int,
        z_channels: int,
        H: int, W: int,
        rank: tuple[int, int],
        tau: Union[float, torch.Tensor] = 1.0,
        df: Union[float, torch.Tensor] = 4.0,
        jitter: float = 1e-6,
        eps: float = 1e-6,
        dropout: float = 0.0,
        backbone_features: int = 256,
        use_mean_shift: bool = False,
        num_modes: int = 4,
    ):
        super().__init__()
        self.h_C, self.H, self.W = h_channels, H, W
        self.S = H * W
        self.D_total = self.h_C * self.S
        self.rank_ch, self.rank_sp = rank
        self.jitter = jitter
        self.eps = eps
        self.use_mean_shift = use_mean_shift
        self.num_modes = num_modes
        
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        self.register_buffer("tau", tau)

        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df, dtype=torch.float32)
        self.log_df = nn.Parameter(torch.log(df))

        self.backbone = nn.Sequential(
            nn.Conv2d(z_channels, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.attention_pool = AttentionPooling(backbone_features)

        if self.use_mean_shift:
            # Learnable mean vectors (M modes)
            self.mean_vectors = nn.ParameterList([
                nn.Parameter(torch.zeros(self.h_C, self.H, self.W))
                for _ in range(self.num_modes)
            ])
            for param in self.mean_vectors:
                nn.init.normal_(param, std=0.01)
            
            # Mixing head predicts weights from global context
            self.mixing_head = nn.Linear(backbone_features, self.num_modes)
            nn.init.zeros_(self.mixing_head.weight)
            nn.init.zeros_(self.mixing_head.bias)
        else:
            # Standard conditional mean prediction
            self.loc_head = nn.Conv2d(backbone_features * 2, self.h_C, 1)
            nn.init.zeros_(self.loc_head.weight)
            nn.init.zeros_(self.loc_head.bias)

        self.sp_D_head = nn.Conv2d(backbone_features * 2, 1, 1)
        self.sp_U_head = nn.Conv2d(backbone_features * 2, self.rank_sp, 1)

        self.ch_D_head = nn.Linear(backbone_features, self.h_C)
        self.ch_U_head = nn.Linear(backbone_features, self.h_C * self.rank_ch)
        
        if not self.use_mean_shift:
            nn.init.zeros_(self.loc_head.weight)
            nn.init.zeros_(self.loc_head.bias)
        
        nn.init.zeros_(self.sp_D_head.weight)
        nn.init.zeros_(self.sp_D_head.bias)
        nn.init.zeros_(self.ch_D_head.weight)
        nn.init.zeros_(self.ch_D_head.bias)
        
        nn.init.normal_(self.sp_U_head.weight, std=1e-4)
        nn.init.zeros_(self.sp_U_head.bias)
        nn.init.normal_(self.ch_U_head.weight, std=1e-4)
        nn.init.zeros_(self.ch_U_head.bias)

    def forward(self, h: torch.Tensor) -> torch.distributions.Distribution:
        B = h.shape[0]
        
        shared_features = self.backbone(h)
        pooled_features = self.attention_pool(shared_features)

        log_ch_D = self.ch_D_head(pooled_features)
        ch_U = self.ch_U_head(pooled_features).view(B, self.h_C, self.rank_ch)

        global_context_map = pooled_features.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        spatial_input = torch.cat([shared_features, global_context_map], dim=1)
        
        if self.use_mean_shift:
            # Predict mixing weights
            logits = self.mixing_head(pooled_features)
            weights = torch.softmax(logits, dim=-1) # [B, M]
            
            # Compute weighted mean: sum(w_k * mu_k)
            means_stack = torch.stack(list(self.mean_vectors), dim=0) # [M, C, H, W]
            loc_shaped = torch.einsum('bm,mchw->bchw', weights, means_stack)
        else:
            # Predict mean from spatial input
            loc = self.loc_head(spatial_input).view(B, -1)
            loc_shaped = loc.view(B, self.h_C, self.H, self.W)

        log_sp_D = self.sp_D_head(spatial_input).view(B, -1)
        sp_U = self.sp_U_head(spatial_input).permute(0, 2, 3, 1).reshape(B, self.S, self.rank_sp)
        
        df = torch.exp(self.log_df) + 2.0

        base_dist = KroneckerProductMVT(
            loc=torch.zeros(B, self.h_C * self.H * self.W, device=h.device, dtype=h.dtype),
            ch_cov=(ch_U, torch.exp(log_ch_D) + self.eps),
            sp_cov=(sp_U, torch.exp(log_sp_D) + self.eps),
            df=df,
            C=self.h_C, H=self.H, W=self.W,
            jitter=self.jitter
        )

        log_s = torch.clamp(self.tau, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(base_dist, loc=loc_shaped, scale=global_scale)
        
        return scaled_dist
        
    def __str__(self):
        df = torch.exp(self.log_df) + 2.0
        mode_str = f", modes={self.num_modes}" if self.use_mean_shift else ""
        return f"ConditionalKPMVTPrior(h_C={self.h_C}, H={self.H}, W={self.W}, rank_ch={self.rank_ch}, rank_sp={self.rank_sp}, tau={self.tau.item():.2f}, df={df.item():.2f}{mode_str})"


class ClassConditionalPrior(nn.Module):
    def __init__(self, priors: List[nn.Module]):
        super(ClassConditionalPrior, self).__init__()
        self.priors = nn.ModuleList(priors)
        self.prior_cls = priors[0].__class__
        self.K = len(priors)
        
        assert all(isinstance(prior, priors[0].__class__) for prior in priors), "All priors must be of the same type"

    def forward(self) -> List[torch.distributions.Distribution]:
        return [prior() for prior in self.priors]

    def __str__(self):
        return f"ClassConditionalPrior(K={self.K}, prior_cls={self.prior_cls.__name__})"


class ConditionalMixturePrior(nn.Module):
    tau: torch.Tensor
    
    def __init__(
        self,
        components: List[nn.Module],
        h_channels: int,
        tau: Union[float, torch.Tensor] = 1.0,
        backbone_features: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.components = nn.ModuleList(components)
        self.K = len(components)
        
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        self.register_buffer("tau", tau)
        
        self.mixing_net = nn.Sequential(
            nn.Conv2d(h_channels, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            AttentionPooling(backbone_features),
            nn.Linear(backbone_features, self.K)
        )

        # Initialize mixing weights to be close to uniform
        nn.init.zeros_(self.mixing_net[-1].weight)
        nn.init.zeros_(self.mixing_net[-1].bias)
        
    def forward(self, h: torch.Tensor, *args, **kwargs) -> torch.distributions.Distribution:
        dists = [comp(*args, **kwargs) for comp in self.components]
        logits = self.mixing_net(h)
        mixture_dist = MixtureDistribution(dists, mixing_logits=logits)
        
        log_s = torch.clamp(self.tau, min=-15.0, max=15.0)
        global_scale = torch.exp(0.5 * log_s)
        scaled_dist = ScaledDistribution(mixture_dist, loc=0.0, scale=global_scale)
        return scaled_dist

    def __str__(self):
        return f"ConditionalMixturePrior(K={self.K}, tau={self.tau.item():.2f})"


# ============================================================================
# Factory Functions
# ============================================================================

def create_unconditional_priors(
    K: int,
    C: int, H: int, W: int,
    *,
    prior_type: str = "kpmvt",
    rank: tuple[int, int] = (8, 8),
    df: float = 4.0,
    init_strategy: str = "simplex_scaled",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
    tau: float = 1.0,
    jitter: float = 1e-6,
    eps: float = 1e-6,
) -> List[nn.Module]:
    """
    Factory to create K unconditional prior components.
    
    Args:
        K: Number of components (classes)
        C: Number of channels
        H: Height
        W: Width
        prior_type: "lrmvn", "kpmvn", or "kpmvt"
        rank: Rank of covariance (int for lrmvn, tuple for kpmvn/kpmvt)
        df: Degrees of freedom (for kpmvt only)
        init_strategy: "zero", "simplex", or "simplex_scaled"
        simplex_scale: Scale of simplex vertices
        noise: Initialization noise
        tau: Global scale parameter
        jitter: Numerical stability jitter
        eps: Small constant for positivity
    
    Returns:
        List of K prior modules
    """
    if prior_type == "lrmvn":
        if isinstance(rank, tuple):
            rank = rank[0]  # Extract single rank value
        params_list = create_lrmvn_priors(
            K, C, H, W, 
            rank=rank, 
            init_strategy=init_strategy,
            simplex_scale=simplex_scale, 
            noise=noise
        )
        priors = []
        for params in params_list:
            tau_val = params.get("tau", tau)
            priors.append(LowRankMVNPrior(
                loc=params["loc"],
                diag=params["diag"],
                U=params["U"],
                tau=tau_val,
                eps=eps
            ))
        return priors
    
    elif prior_type == "kpmvn":
        params_list = create_kpmvn_priors(
            K, C, H, W,
            rank=rank,
            init_strategy=init_strategy,
            simplex_scale=simplex_scale,
            noise=noise
        )
        priors = []
        for params in params_list:
            tau_val = params.get("tau", tau)
            priors.append(KPMVNPrior(
                C=params["C"], H=params["H"], W=params["W"],
                loc=params["loc"],
                cov_ch=params["cov_ch"],
                cov_sp=params["cov_sp"],
                tau=tau_val,
                jitter=jitter,
                eps=eps
            ))
        return priors
    
    elif prior_type == "kpmvt":
        params_list = create_kpmvt_priors(
            K, C, H, W,
            rank=rank,
            df=df,
            init_strategy=init_strategy,
            simplex_scale=simplex_scale,
            noise=noise
        )
        priors = []
        for params in params_list:
            tau_val = params.get("tau", tau)
            priors.append(KPMVTPrior(
                C=params["C"], H=params["H"], W=params["W"],
                loc=params["loc"],
                cov_ch=params["cov_ch"],
                cov_sp=params["cov_sp"],
                tau=tau_val,
                df=params["df"],
                jitter=jitter,
                eps=eps
            ))
        return priors
    
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")


def create_conditional_prior(
    h_channels: int,
    z_channels: int,
    H: int, W: int,
    *,
    prior_type: str = "kpmvt",
    rank: tuple[int, int] = (8, 8),
    df: float = 4.0,
    tau: float = 1.0,
    jitter: float = 1e-6,
    eps: float = 1e-6,
    dropout: float = 0.0,
    backbone_features: int = 256,
    use_mean_shift: bool = False,
    num_modes: int = 4,
) -> nn.Module:
    """
    Factory to create a conditional prior.
    
    Args:
        h_channels: Number of channels in conditioning variable h
        z_channels: Number of channels in latent variable z
        H: Height
        W: Width
        prior_type: "kpmvn" or "kpmvt"
        rank: tuple (rank_ch, rank_sp)
        df: Degrees of freedom (for kpmvt only)
        tau: Global scale parameter
        jitter: Numerical stability jitter
        eps: Small constant for positivity
        dropout: Dropout rate
        backbone_features: Number of features in backbone
        use_mean_shift: Use mixture of means (for kpmvt only)
        num_modes: Number of modes for mean shift
    
    Returns:
        Conditional prior module
    """
    if prior_type == "kpmvn":
        return ConditionalKPMVNPrior(
            h_channels=h_channels,
            z_channels=z_channels,
            H=H, W=W,
            rank=rank,
            tau=tau,
            jitter=jitter,
            eps=eps,
            dropout=dropout,
            backbone_features=backbone_features
        )
    
    elif prior_type == "kpmvt":
        return ConditionalKPMVTPrior(
            h_channels=h_channels,
            z_channels=z_channels,
            H=H, W=W,
            rank=rank,
            tau=tau,
            df=df,
            jitter=jitter,
            eps=eps,
            dropout=dropout,
            backbone_features=backbone_features,
            use_mean_shift=use_mean_shift,
            num_modes=num_modes
        )
    
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")


def create_class_conditional_prior(
    K: int,
    C: int, H: int, W: int,
    **kwargs
) -> ClassConditionalPrior:
    """
    Factory to create a class-conditional prior (wrapper around K unconditional priors).
    
    Args:
        K: Number of classes
        C: Number of channels
        H: Height
        W: Width
        **kwargs: Arguments passed to create_unconditional_priors
    
    Returns:
        ClassConditionalPrior module
    """
    priors = create_unconditional_priors(K, C, H, W, **kwargs)
    return ClassConditionalPrior(priors)


def create_conditional_mixture_prior(
    K: int,
    C: int, H: int, W: int,
    h_channels: int,
    *,
    mixture_tau: float = 1.0,
    backbone_features: int = 256,
    dropout: float = 0.0,
    **component_kwargs
) -> ConditionalMixturePrior:
    """
    Factory to create a conditional mixture prior (K components with mixing network).
    
    Args:
        K: Number of mixture components
        C: Number of channels
        H: Height
        W: Width
        h_channels: Number of channels in conditioning variable h
        mixture_tau: Global scale for mixture
        backbone_features: Number of features in mixing network
        dropout: Dropout rate for mixing network
        **component_kwargs: Arguments passed to create_unconditional_priors
    
    Returns:
        ConditionalMixturePrior module
    """
    components = create_unconditional_priors(K, C, H, W, **component_kwargs)
    return ConditionalMixturePrior(
        components=components,
        h_channels=h_channels,
        tau=mixture_tau,
        backbone_features=backbone_features,
        dropout=dropout
    )


def create_conditional_mixture_of_modes_prior(
    num_modes: int,
    h_channels: int,
    z_channels: int,
    H: int, W: int,
    *,
    prior_type: str = "kpmvt",
    rank: tuple[int, int] = (8, 8),
    df: float = 4.0,
    tau: float = 1.0,
    jitter: float = 1e-6,
    eps: float = 1e-6,
    dropout: float = 0.0,
    backbone_features: int = 256,
    init_strategy: str = "simplex",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
) -> nn.Module:
    """
    Factory to create a conditional prior with mixture of modes (mean shift).
    This uses a single conditional prior with multiple learnable mode vectors.
    
    Args:
        num_modes: Number of mode vectors
        h_channels: Number of channels in conditioning variable h
        z_channels: Number of channels in latent variable z
        H: Height
        W: Width
        prior_type: "kpmvn" or "kpmvt"
        rank: tuple (rank_ch, rank_sp)
        df: Degrees of freedom (for kpmvt only)
        tau: Global scale parameter
        jitter: Numerical stability jitter
        eps: Small constant for positivity
        dropout: Dropout rate
        backbone_features: Number of features in backbone
        init_strategy: "zero" or "simplex" for initializing mode vectors
        simplex_scale: Scale of simplex vertices (used if init_strategy="simplex")
        noise: Standard deviation of initialization noise
    
    Returns:
        Conditional prior module with mixture of modes
    """
    if prior_type == "kpmvt":
        prior = ConditionalKPMVTPrior(
            h_channels=h_channels,
            z_channels=z_channels,
            H=H, W=W,
            rank=rank,
            tau=tau,
            df=df,
            jitter=jitter,
            eps=eps,
            dropout=dropout,
            backbone_features=backbone_features,
            use_mean_shift=True,
            num_modes=num_modes
        )
        
        # Initialize mode vectors with simplex or zeros
        if init_strategy == "zero":
            for param in prior.mean_vectors:
                nn.init.zeros_(param)
                if noise > 0:
                    param.data.add_(torch.randn_like(param) * noise)
        
        elif init_strategy == "simplex":
            from src.utils.prior_init import generate_simplex_means
            # Generate simplex means
            simplex_means = generate_simplex_means(
                num_modes, h_channels, H, W, 
                simplex_scale=simplex_scale, 
                noise=noise
            )
            # Assign to parameters
            for i, param in enumerate(prior.mean_vectors):
                param.data = simplex_means[i].view(h_channels, H, W)
        
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")
        
        return prior
    
    elif prior_type == "kpmvn":
        # KPMVN doesn't support mean shift yet, fall back to regular conditional
        return create_conditional_prior(
            h_channels=h_channels,
            z_channels=z_channels,
            H=H, W=W,
            prior_type=prior_type,
            rank=rank,
            tau=tau,
            jitter=jitter,
            eps=eps,
            dropout=dropout,
            backbone_features=backbone_features
        )
    
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")