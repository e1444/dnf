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
    