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

def kpmvn_simplex_init(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    simplex_scale: float = 1.0,
    noise: float = 0.0,
    tau_marginal: float = 1.0
) -> list[dict]:
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
        simplex_scale: Scale of simplex vertices (distance from origin)
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, C * S) mean vectors initialized at simplex vertices
        ch_cov: tuple (U_ch, D_ch) where U_ch: (K, C, r_ch), D_ch: (K, C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (K, S, r_sp), D_sp: (K, S)
    """
    S = H * W
    D = C * S
    assert C >= K - 1, f"Simplex initialization requires C >= K - 1 (got C={C}, K={K})"
    
    rank_ch, rank_sp = rank
    
    simplex_vertices = torch.zeros(K, K - 1)
    for k in range(K - 1):
        r_k = (1.0 / (2 * (k + 1) * (k + 2)))**0.5
        simplex_vertices[:k+1, k] = -r_k
        simplex_vertices[k+1, k] = (k + 1) * r_k
        
    simplex_vertices = torch.nn.functional.normalize(simplex_vertices, p=2, dim=1)
    
    pixel_scale = simplex_scale / (S**0.5)
    simplex_vertices = simplex_vertices * pixel_scale
    
    loc_ch = torch.zeros(K, C)
    loc_ch[:, :K-1] = simplex_vertices
    
    loc = loc_ch.unsqueeze(-1).expand(K, C, S).contiguous()
    
    loc = loc.view(K, D)
    loc = loc + torch.randn_like(loc) * noise
    
    marg_var = torch.exp(torch.tensor(tau_marginal) / D)
    comp_var = marg_var - (simplex_scale**2 / D)
    comp_var = torch.clamp(comp_var, min=1e-6)
    tau = D * torch.log(comp_var).item()
    
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
            "W": W,
            "tau": tau
        })
    
    return theta


def kpmvt_simplex_init(
    K: int,
    C: int, H: int, W: int,
    *,
    rank: tuple[int, int],
    df: float,
    simplex_scale: float = 1.0,
    noise: float = 0.0,
    tau_marginal: float = 1.0
):
    """
    Initialize Kronecker-Product Multivariate Student T Prior Parameters on Simplex
    
    Args:
        K: Number of components
        C: Number of channels
        H: Height
        W: Width
        rank: tuple (rank_ch, rank_sp)
            rank_ch: Rank of channel covariance
            rank_sp: Rank of spatial covariance
        simplex_scale: Scale of simplex vertices (distance from origin)
        noise: Standard deviation of Gaussian noise added to initialization
    
    Returns:
        loc: (K, C * S) mean vectors initialized at simplex vertices
        ch_cov: tuple (U_ch, D_ch) where U_ch: (K, C, r_ch), D_ch: (K, C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (K, S, r_sp), D_sp: (K, S)
    """
    S = H * W
    D = C * S
    assert C >= K - 1, f"Simplex initialization requires C >= K - 1 (got C={C}, K={K})"
    
    rank_ch, rank_sp = rank
    
    simplex_vertices = torch.zeros(K, K - 1)
    for k in range(K - 1):
        r_k = (1.0 / (2 * (k + 1) * (k + 2)))**0.5
        simplex_vertices[:k+1, k] = -r_k
        simplex_vertices[k+1, k] = (k + 1) * r_k
        
    simplex_vertices = torch.nn.functional.normalize(simplex_vertices, p=2, dim=1)
    
    pixel_scale = simplex_scale / (S**0.5)
    simplex_vertices = simplex_vertices * pixel_scale
    
    loc_ch = torch.zeros(K, C)
    loc_ch[:, :K-1] = simplex_vertices
    
    loc = loc_ch.unsqueeze(-1).expand(K, C, S).contiguous()
    
    loc = loc.view(K, D)
    loc = loc + torch.randn_like(loc) * noise
    nu = df  # degrees of freedom used by MVT

    inflation = nu / (nu - 2)
    marg_var = torch.exp(torch.tensor(tau_marginal) / D) / inflation
    comp_var = marg_var - (simplex_scale**2 / D)
    comp_var = torch.clamp(comp_var, min=1e-6)
    tau = D * torch.log(comp_var).item()
    
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
            "df": df,
            "C": C,
            "H": H,
            "W": W,
            "tau": tau
        })
    
    return theta
    