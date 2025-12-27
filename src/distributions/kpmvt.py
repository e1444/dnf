import math
import torch
from torch.distributions import Distribution, Chi2
from torch.distributions.utils import _standard_normal

from src.distributions.kpmvn import KroneckerProductMVN


class KroneckerProductMVT(Distribution):
    """
    Multivariate Student T with covariance Sigma = Sigma_ch ⊗ Sigma_sp.
    Supports arbitrary batch shapes for loc, ch_cov, and sp_cov.
    """
    arg_constraints = {}    # type: ignore
    has_rsample = True
    
    def __init__(
        self,
        loc,
        ch_cov,
        sp_cov,
        df,
        C, H, W,
        jitter=1e-6
    ):
        """
        loc: (..., C * S) mean vector
        ch_cov: tuple (U_ch, D_ch) where U_ch: (..., C, r_ch), D_ch: (..., C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (..., S, r_sp), D_sp: (..., S)
        df: degrees of freedom
        """
        self.base_dist = KroneckerProductMVN(
            loc, ch_cov, sp_cov, C, H, W, jitter
        )
        # Ensure df is a tensor on the correct device
        self.df = torch.as_tensor(df, dtype=loc.dtype, device=loc.device)
        
        super().__init__(self.base_dist.batch_shape, self.base_dist.event_shape)
        
    def log_prob(self, value):
        # 1. Compute Mahalanobis distance squared
        # Use the direct method from base_dist for stability and correctness
        mahalanobis_sq = self.base_dist.mahalanobis_sq(value)
        
        # Clamp for numerical safety (prevent negative values due to float errors)
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0)
        
        D = self.base_dist.D_ch * self.base_dist.D_sp
        log_det = self.base_dist.log_det_total
        
        # 2. Compute MVT log prob
        # log_p = lgamma((nu+p)/2) - lgamma(nu/2) - (p/2)log(nu*pi) - 0.5*log_det - (nu+p)/2 * log(1 + mahalanobis_sq/nu)
        
        nu = self.df
        p = D
        
        term1 = torch.lgamma(0.5 * (nu + p))
        term2 = torch.lgamma(0.5 * nu)
        term3 = 0.5 * p * torch.log(nu * math.pi)
        term4 = 0.5 * log_det
        term5 = 0.5 * (nu + p) * torch.log1p(mahalanobis_sq / nu)
        
        return term1 - term2 - term3 - term4 - term5

    def rsample(self, sample_shape=torch.Size()):
        # X = mu + Z * sqrt(nu / U)
        # Z ~ N(0, Sigma)
        # U ~ Chi2(nu)
        
        # 1. Sample Y ~ N(mu, Sigma) from base dist
        Y = self.base_dist.rsample(sample_shape)
        
        # 2. Get centered Z
        Z = Y - self.base_dist.loc
        
        # 3. Sample U ~ Chi2(nu)
        # Expand df to match batch_shape
        nu = self.df.expand(self.batch_shape)
        chi2 = Chi2(nu)
        U = chi2.rsample(sample_shape) # (Sample..., Batch...)
        
        # 4. Compute scale factor sqrt(nu / U)
        # Clamp U to prevent division by zero or explosion
        U_clamped = torch.clamp(U, min=1e-8)
        scale = torch.sqrt(nu / U_clamped)
        
        # Unsqueeze scale to match event dimensions (C, H, W)
        # Z shape: (Sample..., Batch..., C, H, W)
        # scale shape: (Sample..., Batch...) -> (Sample..., Batch..., 1, 1, 1)
        for _ in range(len(self.event_shape)):
            scale = scale.unsqueeze(-1)
            
        return self.base_dist.loc + Z * scale
    
    def anisotropy_loss(self):
        return self.base_dist.anisotropy_loss()