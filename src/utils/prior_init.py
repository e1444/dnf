import torch


def lrmvn_simplex_init(K: int, D: int, rank: int, simplex_scale: float = 1.0, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize Low-Rank Multivariate Normal Prior Parameters on Simplex
    
    Args:
        K: Number of components
        D: Latent dimensionality
    
    Returns:
        loc: (K, D) mean vectors initialized at 0
        diag: (K, D) diagonal covariance initialized at 1
        U: (K, D, D) low-rank factors initialized at 0
    """
    # NOTE: Can be optimized to D >= K - 1 only
    assert D >= K, "Simplex initialization requires D >= K"
    loc = torch.zeros(K, D)
    for i in range(K):
        loc[i, i] = simplex_scale
    loc = loc + torch.randn_like(loc) * noise
    diag = torch.ones(K, D)
    U = torch.zeros(K, D, rank)
    
    return loc, diag, U