import torch
from typing import List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.priors import LowRankMVNPrior, KPMVNPrior, KPMVTPrior


def generate_simplex_means(K: int, C: int, H: int, W: int, simplex_scale: float = 1.0, noise: float = 0.0) -> torch.Tensor:
    """
    Generate K mean vectors arranged on a (K-1)-dimensional simplex in C*H*W space.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        simplex_scale: Scale of simplex vertices (distance from origin)
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, C*H*W) mean vectors
    """
    S = H * W
    D = C * S
    
    if simplex_scale == 0.0:
        # Zero initialization
        loc = torch.zeros(K, D)
    else:
        # Simplex initialization
        assert C >= K - 1, f"Simplex initialization requires C >= K - 1 (got C={C}, K={K})"
        
        # Generate simplex vertices in (K-1)-dimensional space
        simplex_vertices = torch.zeros(K, K - 1)
        for k in range(K - 1):
            r_k = (1.0 / (2 * (k + 1) * (k + 2)))**0.5
            simplex_vertices[:k+1, k] = -r_k
            simplex_vertices[k+1, k] = (k + 1) * r_k
        
        simplex_vertices = torch.nn.functional.normalize(simplex_vertices, p=2, dim=1)
        
        # Scale by pixel-wise scale (divide by sqrt(S) to maintain unit norm per pixel)
        pixel_scale = simplex_scale / (S**0.5)
        simplex_vertices = simplex_vertices * pixel_scale
        
        # Embed in channel space (first K-1 channels)
        loc_ch = torch.zeros(K, C)
        loc_ch[:, :K-1] = simplex_vertices
        
        # Broadcast across spatial dimensions
        loc = loc_ch.unsqueeze(-1).expand(K, C, S).contiguous()
        loc = loc.view(K, D)
    
    # Add noise
    if noise > 0.0:
        loc = loc + torch.randn_like(loc) * noise
    
    return loc


def create_lrmvn_priors(
    K: int, 
    C: int, H: int, W: int, 
    *, 
    rank: int, 
    init_strategy: str = "zero",
    simplex_scale: float = 1.0, 
    noise: float = 0.0,
    tau: float = 0.0,
    eps: float = 1e-6
) -> List['LowRankMVNPrior']:
    """
    Factory function to create K LRMVN prior modules.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: Rank of low-rank covariance
        init_strategy: "zero" or "simplex"
        simplex_scale: Scale of simplex vertices (used if init_strategy="simplex")
        noise: Standard deviation of Gaussian noise
        tau: Global scale parameter
        eps: Small constant for numerical stability
    
    Returns:
        List of K LowRankMVNPrior modules
    """
    from src.models.priors import LowRankMVNPrior
    
    D = C * H * W
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")
    
    priors = []
    for k in range(K):
        priors.append(LowRankMVNPrior(
            loc=means[k],
            diag=torch.ones(D),
            U=torch.zeros(D, rank),
            tau=torch.tensor(tau) if not isinstance(tau, torch.Tensor) else tau,
            eps=eps
        ))
    return priors


def create_kpmvn_priors(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    init_strategy: str = "zero",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
    tau: float = 0.0,
    jitter: float = 1e-6,
    eps: float = 1e-6,
) -> List['KPMVNPrior']:
    """
    Factory function to create K KPMVN prior modules.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
        init_strategy: "zero" or "simplex" or "simplex_scaled"
        simplex_scale: Scale of simplex vertices (used if init_strategy contains "simplex")
        noise: Standard deviation of Gaussian noise
        tau: Global scale parameter (will be overridden in simplex_scaled)
        jitter: Numerical stability jitter
        eps: Small constant for positivity
    
    Returns:
        List of K KPMVNPrior modules
    """
    from src.models.priors import KPMVNPrior
    
    D = C * H * W
    S = H * W
    rank_ch, rank_sp = rank
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
        tau_values = [tau] * K
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        tau_values = [tau] * K
    elif init_strategy == "simplex_scaled":
        # Simplex with variance adjustment to maintain unit marginal variance
        # For Gaussian: marginal_var = component_var + mean_var
        # Want: component_var + (simplex_scale^2/D) = 1.0
        # Therefore: component_var = 1.0 - simplex_scale^2/D
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        comp_var = torch.tensor(1.0 - (simplex_scale**2 / D))
        comp_var = torch.clamp(comp_var, min=1e-6)
        tau_values = [torch.log(comp_var).item()] * K
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")
    
    priors = []
    for k in range(K):
        priors.append(KPMVNPrior(
            C=C, H=H, W=W,
            loc=means[k],
            cov_ch=(torch.zeros(C, rank_ch), torch.ones(C)),
            cov_sp=(torch.zeros(S, rank_sp), torch.ones(S)),
            tau=tau_values[k],
            jitter=jitter,
            eps=eps
        ))
    return priors


def create_kpmvt_priors(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    df: float,
    init_strategy: str = "zero",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
    tau: Optional[float] = None,
    jitter: float = 1e-6,
    eps: float = 1e-6,
) -> List['KPMVTPrior']:
    """
    Factory function to create K KPMVT prior modules.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
        df: Degrees of freedom
        init_strategy: "zero" or "simplex" or "simplex_scaled"
        simplex_scale: Scale of simplex vertices (used if init_strategy contains "simplex")
        noise: Standard deviation of Gaussian noise
        tau: Global scale parameter (will be overridden in simplex_scaled, auto-computed if None)
        jitter: Numerical stability jitter
        eps: Small constant for positivity
    
    Returns:
        List of K KPMVTPrior modules
    """
    from src.models.priors import KPMVTPrior
    
    # Validate df
    assert df > 2.0, f"Student-T requires df > 2 for finite variance (got df={df})"
    
    D = C * H * W
    S = H * W
    rank_ch, rank_sp = rank
    
    # Compute default tau if not provided
    if tau is None:
        tau = torch.log(torch.tensor((df - 2.0) / df)).item()
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
        tau_values = [tau] * K
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        tau_values = [tau] * K
    elif init_strategy == "simplex_scaled":
        # Simplex with variance adjustment to maintain unit marginal variance
        # For Student-T: marginal_var = inflation * component_var + mean_var
        # Want: inflation * component_var + (simplex_scale^2/D) = 1.0
        # Therefore: component_var = (1.0 - simplex_scale^2/D) / inflation
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        nu = df
        inflation = nu / (nu - 2)
        comp_var = torch.tensor((1.0 - simplex_scale**2 / D) / inflation)
        comp_var = torch.clamp(comp_var, min=1e-6)
        tau_values = [torch.log(comp_var).item()] * K
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")
    
    priors = []
    for k in range(K):
        priors.append(KPMVTPrior(
            C=C, H=H, W=W,
            loc=means[k],
            cov_ch=(torch.zeros(C, rank_ch), torch.ones(C)),
            cov_sp=(torch.zeros(S, rank_sp), torch.ones(S)),
            tau=tau_values[k],
            df=df,
            jitter=jitter,
            eps=eps
        ))
    return priors


# ============================================================================
# High-level Factory Functions
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
    tau: Optional[float] = None,
    jitter: float = 1e-6,
    eps: float = 1e-6,
) -> List:
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
        tau: Global scale parameter (if None, computed based on prior_type and df)
        jitter: Numerical stability jitter
        eps: Small constant for positivity
    
    Returns:
        List of K prior modules
    """
    # Validate parameters
    if prior_type == "kpmvt" and df <= 2.0:
        raise ValueError(f"Student-T requires df > 2 for finite variance (got df={df})")
    
    # Compute default tau if not provided
    if tau is None:
        if prior_type == "kpmvt":
            # Student-T: tau = log((df-2)/df) to match ActNorm variance=1
            tau = torch.log(torch.tensor((df - 2.0) / df)).item()
        else:
            # Gaussian: natural variance=1, so tau=0
            tau = 0.0
    
    if prior_type == "lrmvn":
        rank_val = rank[0] if isinstance(rank, tuple) else rank
        return create_lrmvn_priors(
            K, C, H, W, 
            rank=rank_val, 
            init_strategy=init_strategy,
            simplex_scale=simplex_scale, 
            noise=noise,
            tau=tau,
            eps=eps
        )
    
    elif prior_type == "kpmvn":
        return create_kpmvn_priors(
            K, C, H, W,
            rank=rank,
            init_strategy=init_strategy,
            simplex_scale=simplex_scale,
            noise=noise,
            tau=tau,
            jitter=jitter,
            eps=eps
        )
    
    elif prior_type == "kpmvt":
        return create_kpmvt_priors(
            K, C, H, W,
            rank=rank,
            df=df,
            init_strategy=init_strategy,
            simplex_scale=simplex_scale,
            noise=noise,
            tau=tau,
            jitter=jitter,
            eps=eps
        )
    
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
    class_conditional_mean_shift: bool = False,
    num_classes: Optional[int] = None,
    init_strategy: str = "zero",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
):
    """
    Factory to create a conditional prior p(z | h).
    
    In the VS-Flow hierarchy: p(z_struct^(i) | h^(i+1))
    
    Args:
        h_channels: Number of channels in conditioning variable h^(i+1) (from next level)
        z_channels: Number of channels in latent variable z_struct^(i) being modeled
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
        init_strategy: "zero" or "simplex" for initializing mode vectors
        simplex_scale: Scale of simplex vertices (used if init_strategy="simplex")
        noise: Standard deviation of initialization noise
    
    Returns:
        Conditional prior module
    """
    from src.models.priors import ConditionalKPMVNPrior, ConditionalKPMVTPrior
    
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
        if class_conditional_mean_shift and not use_mean_shift:
            raise ValueError("class_conditional_mean_shift requires use_mean_shift=True")
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
            use_mean_shift=use_mean_shift,
            num_modes=num_modes,
            class_conditional_mean_shift=class_conditional_mean_shift,
            num_classes=num_classes,
        )
        
        # Initialize mode vectors if using mean shift
        if use_mean_shift:
            # Generate simplex means using shared utility
            simplex_means = generate_simplex_means(
                num_modes, h_channels, H, W, 
                simplex_scale=simplex_scale if init_strategy == "simplex" else 0.0, 
                noise=noise
            )
            # Assign to parameters
            for i, param in enumerate(prior.mean_vectors):
                param.data = simplex_means[i].view(h_channels, H, W)
        
        return prior
    
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")


def create_class_conditional_prior(
    K: int,
    C: int, H: int, W: int,
    **kwargs
):
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
    from src.models.priors import ClassConditionalPrior
    
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
    components: Optional[dict] = None,
    **legacy_kwargs
):
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
        components: Dict of parameters for unconditional component priors
        **legacy_kwargs: Fallback for old flat structure (deprecated)
    
    Returns:
        ConditionalMixturePrior module
    """
    from src.models.priors import ConditionalMixturePrior
    
    # Use nested components dict if provided, otherwise fall back to flat structure
    if components is not None:
        component_kwargs = components
    else:
        # Legacy fallback: filter valid unconditional keys from flat structure
        valid_unconditional_keys = {
            'prior_type', 'rank', 'df', 'init_strategy', 'simplex_scale', 
            'noise', 'tau', 'jitter', 'eps'
        }
        component_kwargs = {k: v for k, v in legacy_kwargs.items() if k in valid_unconditional_keys}
    
    component_priors = create_unconditional_priors(K, C, H, W, **component_kwargs)
    return ConditionalMixturePrior(
        components=component_priors,
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
):
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
    from src.models.priors import ConditionalKPMVTPrior
    import torch.nn as nn
    
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
