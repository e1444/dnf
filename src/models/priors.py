import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal

from src.distributions.kpmvn import KroneckerProductMVN


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
        
        assert diag.positive().all(), "Diagonal covariance must be positive"
        
        self.loc = nn.Parameter(loc)
        self.log_diag = nn.Parameter(torch.log(diag))
        self.U = nn.Parameter(U)

    def forward(self, cov_scale: float = 1.0, eps: float = 1e-6):
        return LowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.U * (cov_scale ** 0.5),
            cov_diag=torch.exp(self.log_diag) * cov_scale + eps
        )
        
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
        
        assert ch_cov[1].positive().all(), "Channel diagonal must be positive"
        assert sp_cov[1].positive().all(), "Spatial diagonal must be positive"

        self._loc = nn.Parameter(loc)
        self.ch_cov_factor = nn.Parameter(ch_cov[0])
        self.log_ch_cov_diag = nn.Parameter(torch.log(ch_cov[1]))
        self.sp_cov_factor = nn.Parameter(sp_cov[0])
        self.log_sp_cov_diag = nn.Parameter(torch.log(sp_cov[1]))

        self.jitter = jitter
        
    def forward(self, ch_cov_scale: float = 1.0, sp_cov_scale: float = 1.0, eps: float = 1e-6):
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
                dist = prior()
            
            distributions.append(dist)
            
        return distributions