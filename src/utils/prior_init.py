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

def kpmvn_simplex_init(K: int, C: int, H: int, W: int, *, rank_ch: int, rank_sp: int, simplex_scale: float = 1.0, noise: float = 0.0) -> list[dict]:
    """
    Initialize Kronecker-Product Multivariate Normal Prior Parameters on Simplex
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
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
            "ch_cov": (U_ch[k], D_ch[k]),
            "sp_cov": (U_sp[k], D_sp[k]),
            "C": C,
            "H": H,
            "W": W
        })
    
    return theta