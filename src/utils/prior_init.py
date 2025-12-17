import torch


def lrmvn_simplex_init(K: int, C: int, H: int, W: int, *, rank: int, simplex_scale: float = 1.0, noise: float = 0.0) -> list[dict]:
    """
    Initialize Low-Rank Multivariate Normal Prior Parameters on Simplex
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: Rank of low-rank covariance
        simplex_scale: Scale of simplex vertices
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, D) mean vectors initialized at 0
        diag: (K, D) diagonal covariance initialized at 1
        U: (K, D, D) low-rank factors initialized at 0
    """
    # NOTE: Can be optimized to D >= K - 1 only
    D = C * H * W
    assert D >= K, "Simplex initialization requires D >= K"
    
    loc = torch.zeros(K, D)
    for i in range(K):
        loc[i, i] = simplex_scale
    loc = loc + torch.randn_like(loc) * noise
    diag = torch.ones(K, D)
    U = torch.zeros(K, D, rank)
    
    theta = []
    for k in range(K):
        theta.append({
            "loc": loc[k],
            "diag": diag[k],
            "U": U[k]
        })
    
    return theta

def kpmvn_zero_init(K: int, C: int, H: int, W: int, *, rank: tuple[int, int], noise: float = 0.0) -> list[dict]:
    """
    Initialize Kronecker-Product Multivariate Normal Prior Parameters at Zero
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
            rank_ch: Rank of channel covariance
            rank_sp: Rank of spatial covariance
        simplex_scale: Scale of simplex vertices
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, C * S) mean vectors initialized at zero
        ch_cov: tuple (U_ch, D_ch) where U_ch: (K, C, r_ch), D_ch: (K, C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (K, S, r_sp), D_sp: (K, S)
    """
    S = H * W
    D = C * S
    rank_ch, rank_sp = rank
    
    loc = torch.zeros(K, C, S)
    loc = loc.view(K, D)
    loc = loc + torch.randn_like(loc) * noise
    
    U_ch = torch.zeros(K, C, rank_ch)
    D_ch = torch.ones(K, C)
    
    U_sp = torch.zeros(K, S, rank_sp)
    D_sp = torch.ones(K, S)
    
    theta = []
    for k in range(K):
        theta.append({
            "loc": loc[k],
            "cov_ch": (U_ch[k], D_ch[k]),
            "cov_sp": (U_sp[k], D_sp[k]),
            "C": C,
            "H": H,
            "W": W
        })
    
    return theta

def kpmvn_random_init(K: int, C: int, H: int, W: int, *, rank: tuple[int, int], scale: float = 1.0, noise: float = 0.0) -> list[dict]:
    """
    Initialize Kronecker-Product Multivariate Normal Prior Parameters with Random Directions.
    Used when C < K, where orthogonal simplex initialization is impossible.
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
        scale: Magnitude of the mean vectors
        noise: Standard deviation of Gaussian noise added to initialization
    """
    S = H * W
    D = C * S
    rank_ch, rank_sp = rank
    
    # Generate random directions in channel space
    channel_means = torch.randn(K, C)
    # Normalize to unit length
    channel_means = torch.nn.functional.normalize(channel_means, p=2, dim=1)
    # Scale to target magnitude (matching simplex energy: scale / sqrt(S))
    channel_means = channel_means * (scale / (S**0.5))
    
    loc = torch.zeros(K, C, S)
    # Broadcast spatially: (K, C) -> (K, C, S)
    for i in range(K):
        loc[i, :, :] = channel_means[i].view(C, 1).expand(C, S)
    
    loc = loc.view(K, D)
    loc = loc + torch.randn_like(loc) * noise
    
    U_ch = torch.zeros(K, C, rank_ch)
    D_ch = torch.ones(K, C)
    
    U_sp = torch.zeros(K, S, rank_sp)
    D_sp = torch.ones(K, S)
    
    theta = []
    for k in range(K):
        theta.append({
            "loc": loc[k],
            "cov_ch": (U_ch[k], D_ch[k]),
            "cov_sp": (U_sp[k], D_sp[k]),
            "C": C,
            "H": H,
            "W": W
        })
    
    return theta

def kpmvn_simplex_init(K: int, C: int, H: int, W: int, *, rank: tuple[int, int], simplex_scale: float = 1.0, noise: float = 0.0) -> list[dict]:
    """
    Initialize Kronecker-Product Multivariate Normal Prior Parameters on Simplex
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
            rank_ch: Rank of channel covariance
            rank_sp: Rank of spatial covariance
        simplex_scale: Scale of simplex vertices
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, C * S) mean vectors initialized at simplex vertices
        ch_cov: tuple (U_ch, D_ch) where U_ch: (K, C, r_ch), D_ch: (K, C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (K, S, r_sp), D_sp: (K, S)
    """
    S = H * W
    D = C * S
    assert C >= K, "Channel-based initialization requires C >= K"
    
    rank_ch, rank_sp = rank
    
    loc = torch.zeros(K, C, S)
    for i in range(K):
        loc[i, i, :] = simplex_scale / (S**0.5)
    
    loc = loc.view(K, D)
    loc = loc + torch.randn_like(loc) * noise
    
    U_ch = torch.zeros(K, C, rank_ch)
    D_ch = torch.ones(K, C)
    
    U_sp = torch.zeros(K, S, rank_sp)
    D_sp = torch.ones(K, S)
    
    theta = []
    for k in range(K):
        theta.append({
            "loc": loc[k],
            "cov_ch": (U_ch[k], D_ch[k]),
            "cov_sp": (U_sp[k], D_sp[k]),
            "C": C,
            "H": H,
            "W": W
        })
    
    return theta