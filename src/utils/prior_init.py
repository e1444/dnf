import torch
from copy import deepcopy


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


def lrmvn_template(C: int, H: int, W: int, *, rank: int) -> dict:
    """
    Create a single LRMVN component template with zero mean.
    
    Args:
        C: Number of channels
        H: Height
        W: Width
        rank: Rank of low-rank covariance
    
    Returns:
        Parameter dict with loc, diag, and U
    """
    D = C * H * W
    return {
        "loc": torch.zeros(D),
        "diag": torch.ones(D),
        "U": torch.zeros(D, rank)
    }


def create_lrmvn_priors(
    K: int, 
    C: int, H: int, W: int, 
    *, 
    rank: int, 
    init_strategy: str = "zero",
    simplex_scale: float = 1.0, 
    noise: float = 0.0
) -> list[dict]:
    """
    Factory function to create K LRMVN prior components.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: Rank of low-rank covariance
        init_strategy: "zero" or "simplex"
        simplex_scale: Scale of simplex vertices (used if init_strategy="simplex")
        noise: Standard deviation of Gaussian noise
    
    Returns:
        List of K parameter dicts
    """
    template = lrmvn_template(C, H, W, rank=rank)
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")
    
    return [{**deepcopy(template), "loc": means[k]} for k in range(K)]


def kpmvn_template(C: int, H: int, W: int, *, rank: tuple[int, int]) -> dict:
    """
    Create a single KPMVN component template with zero mean.
    
    Args:
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
    
    Returns:
        Parameter dict with loc, cov_ch, cov_sp
    """
    S = H * W
    D = C * S
    rank_ch, rank_sp = rank
    
    return {
        "loc": torch.zeros(D),
        "cov_ch": (torch.zeros(C, rank_ch), torch.ones(C)),
        "cov_sp": (torch.zeros(S, rank_sp), torch.ones(S)),
        "C": C,
        "H": H,
        "W": W
    }


def create_kpmvn_priors(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    init_strategy: str = "zero",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
) -> list[dict]:
    """
    Factory function to create K KPMVN prior components.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
        init_strategy: "zero" or "simplex" or "simplex_scaled"
        simplex_scale: Scale of simplex vertices (used if init_strategy contains "simplex")
        noise: Standard deviation of Gaussian noise
    
    Returns:
        List of K parameter dicts
    """
    D = C * H * W
    template = kpmvn_template(C, H, W, rank=rank)
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
        return [{**deepcopy(template), "loc": means[k]} for k in range(K)]
    
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        return [{**deepcopy(template), "loc": means[k]} for k in range(K)]
    
    elif init_strategy == "simplex_scaled":
        # Simplex with variance adjustment to maintain unit marginal variance
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        comp_var = torch.tensor(1.0) - (simplex_scale**2 / D)
        comp_var = torch.clamp(comp_var, min=1e-6)
        return [{**deepcopy(template), "loc": means[k], "tau": torch.log(comp_var)} for k in range(K)]
    
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")


def kpmvt_template(C: int, H: int, W: int, *, rank: tuple[int, int], df: float) -> dict:
    """
    Create a single KPMVT component template with zero mean.
    
    Args:
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
        df: Degrees of freedom
    
    Returns:
        Parameter dict with loc, cov_ch, cov_sp, df
    """
    S = H * W
    D = C * S
    rank_ch, rank_sp = rank
    
    return {
        "loc": torch.zeros(D),
        "cov_ch": (torch.zeros(C, rank_ch), torch.ones(C)),
        "cov_sp": (torch.zeros(S, rank_sp), torch.ones(S)),
        "df": df,
        "C": C,
        "H": H,
        "W": W
    }


def create_kpmvt_priors(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    df: float,
    init_strategy: str = "zero",
    simplex_scale: float = 1.0,
    noise: float = 0.0,
) -> list[dict]:
    """
    Factory function to create K KPMVT prior components.
    
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
    
    Returns:
        List of K parameter dicts
    """
    D = C * H * W
    template = kpmvt_template(C, H, W, rank=rank, df=df)
    
    if init_strategy == "zero":
        means = generate_simplex_means(K, C, H, W, simplex_scale=0.0, noise=noise)
        return [{**deepcopy(template), "loc": means[k]} for k in range(K)]
    
    elif init_strategy == "simplex":
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        return [{**deepcopy(template), "loc": means[k]} for k in range(K)]
    
    elif init_strategy == "simplex_scaled":
        # Simplex with variance adjustment for Student-T
        means = generate_simplex_means(K, C, H, W, simplex_scale=simplex_scale, noise=noise)
        nu = df
        inflation = nu / (nu - 2)
        marg_var = torch.tensor(1.0) / inflation
        comp_var = marg_var - (simplex_scale**2 / D)
        comp_var = torch.clamp(comp_var, min=1e-6)
        return [{**deepcopy(template), "loc": means[k], "tau": torch.log(comp_var)} for k in range(K)]
    
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")
    