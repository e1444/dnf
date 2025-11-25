import torch
import torch.nn.functional as F

class MultivariateLowRankStudentT(torch.distributions.Distribution):
    def __init__(self, mu: torch.Tensor, v: torch.Tensor, U: torch.Tensor, df: torch.Tensor, eps: float = 1e-6):
        self.mu = mu
        self._v = v
        self.U = U
        self._df = df
        self._eps = eps
        
        self.cov_diag = self.v
        
        self.mvn = torch.distributions.LowRankMultivariateNormal(
            loc=torch.zeros_like(self.mu),
            cov_factor = self.U,
            cov_diag=self.cov_diag
        )
        self.chi2 = torch.distributions.Chi2(df=df)
        
    def sample(self, sample_shape=torch.Size()):
        z = self.mvn.sample(sample_shape)
        s = self.chi2.sample(sample_shape)
        scale = torch.sqrt(self.df / s).unsqueeze(-1)
        return self.mu + scale * z
    
    def log_prob(self, value):
        y = value - self.mu
        d = y.shape[-1]
        df = self.df

        # Diagonal part
        D = self.cov_diag  # (d,)
        Dinv = 1.0 / D

        # y^T D^{-1} y
        Dy = Dinv * y
        term1 = torch.sum(y * Dy, dim=-1)  # scalar or batch

        # Low-rank part: woodbury
        # A = I + U^T D^{-1} U   (r × r)
        UDinv = self.U * Dinv.unsqueeze(-1)       # (d × r)
        A = torch.eye(self.U.shape[1], device=value.device, dtype=value.dtype) + \
            self.U.transpose(-1, -2) @ UDinv

        # Solve A^{-1} (U^T D^{-1} y)
        b = self.U.transpose(-1, -2) @ Dy.unsqueeze(-1)  # (r,1)
        Ainv_b = torch.linalg.solve(A, b).squeeze(-1)    # (r,)

        # y^T D^{-1} U A^{-1} U^T D^{-1} y
        term2 = torch.sum(b.squeeze(-1) * Ainv_b, dim=-1)

        # Mahalanobis
        M = term1 - term2

        # log|Σ| = log|D| + log|I + U^T D^{-1}U|
        logdet_D = torch.sum(torch.log(D))
        logdet = logdet_D + torch.logdet(A)

        # Student-t log density
        c1 = torch.lgamma((df + d) / 2) - torch.lgamma(df / 2)
        c2 = -0.5 * (d * (torch.log(df) + torch.log(torch.tensor(torch.pi, device=value.device, dtype=value.dtype))) + logdet)
        c3 = -0.5 * (df + d) * torch.log1p(M / df)

        return c1 + c2 + c3
    
    @property
    def v(self):
        return F.softplus(self._v) + self._eps  # Ensure positivity
    
    @property
    def df(self):
        return F.softplus(self._df) + 2 + self._eps  # Ensure > 2 for finite variance
        

def get_target_distributions(mu, v, U, df, num_classes, eps=1e-4, device=torch.device('cpu')):
    """Creates a list of target distributions."""
    return [
        MultivariateLowRankStudentT(
            mu=mu[i],
            v=v[i],
            U=U[i],
            df=df[i],
            eps=eps
        ) for i in range(num_classes)
    ]
    