import torch
from torch.distributions import Distribution
from torch.distributions.utils import _standard_normal


class KroneckerProductMVN(Distribution):
    """
    Multivariate Normal with covariance Sigma = Sigma_ch ⊗ Sigma_sp,
    where Sigma_ch = D_ch + U_ch U_ch^T (C x C) and
          Sigma_sp = D_sp + U_sp U_sp^T (S x S) with S = H*W.

    Optimizations:
      - log-determinant via matrix-determinant-lemma
      - inverse application via Woodbury (precompute small r×r Cholesky)
      - rsample: optimized low-rank sampler that never materializes S×S matrices
    """
    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, ch_cov, sp_cov, C, H, W, jitter=1e-6):
        """
        loc: (C * S,) mean vector
        ch_cov: tuple (U_ch, D_ch) where U_ch: (C, r_ch), D_ch: (C,)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (S, r_sp), D_sp: (S,)
        """
        self.C, self.H, self.W = C, H, W
        self.D_ch = C
        self.D_sp = H * W

        self._loc = loc
        self.ch_cov_factor, self.ch_cov_diag = ch_cov
        self.sp_cov_factor, self.sp_cov_diag = sp_cov

        self.jitter = jitter

        # Precompute log-determinants
        self._log_det_ch = self._compute_log_det(self.ch_cov_diag, self.ch_cov_factor)
        self._log_det_sp = self._compute_log_det(self.sp_cov_diag, self.sp_cov_factor)
        self.log_det_total = self.D_sp * self._log_det_ch + self.D_ch * self._log_det_sp

        # Precompute Woodbury caches (for inverse application)
        self.ch_woodbury_cache = self._precompute_woodbury(self.ch_cov_diag, self.ch_cov_factor, jitter)
        self.sp_woodbury_cache = self._precompute_woodbury(self.sp_cov_diag, self.sp_cov_factor, jitter)

        super().__init__(batch_shape=torch.Size(), event_shape=loc.shape)

    @staticmethod
    def _compute_log_det(cov_diag, cov_factor):
        # cov_diag: (D,), cov_factor: (D, r)
        r = cov_factor.shape[1]
        D_inv = 1.0 / cov_diag
        # M = I_r + U^T D^{-1} U
        U_scaled = cov_factor * D_inv.unsqueeze(-1)  # (D, r)
        M = torch.eye(r, device=cov_diag.device, dtype=cov_diag.dtype) + cov_factor.t() @ U_scaled
        sign, logabsdet_M = torch.linalg.slogdet(M)
        logdet_D = torch.sum(torch.log(cov_diag))
        return logdet_D + logabsdet_M

    def _precompute_woodbury(self, D, U, jitter):
        """
        Precompute D_inv, Cholesky of (I + U^T D^{-1} U), and U.
        D: (D,), U: (D, r)
        Returns dict with: "D_inv": (D,), "L_inner": (r, r), "U": (D, r)
        """
        r = U.shape[1]
        D_inv = 1.0 / D
        # M_inner = I + U^T D^{-1} U
        M_inner = torch.eye(r, device=D.device, dtype=D.dtype) + U.t() @ (U * D_inv.unsqueeze(-1))
        # add jitter for numerical stability
        M_inner = M_inner + jitter * torch.eye(r, device=D.device, dtype=D.dtype)
        L_inner = torch.linalg.cholesky(M_inner)  # (r, r), lower triangular
        return {"D_inv": D_inv, "L_inner": L_inner, "U": U}

    def _apply_woodbury_inverse(self, M, cache):
        """
        Applies (D + U U^T)^{-1} @ M where M's last dim has size D.
        M can have arbitrary leading batch dims.
        cache: from _precompute_woodbury
        Returns tensor with same shape as M.
        """
        D_inv = cache["D_inv"]              # (D,)
        L_inner = cache["L_inner"]          # (r, r) (cholesky)
        U = cache["U"]                      # (D, r)

        D = D_inv.shape[0]
        assert M.shape[-1] == D, "Last dim of M must equal covariance dimension"

        # Broadcast D_inv to multiply last dim: shape (1,1,..., D)
        shape_for_D = [1] * (M.dim() - 1) + [-1]
        D_inv_shaped = D_inv.view(*shape_for_D)

        # D^{-1} * M  (broadcasted)
        D_inv_M = M * D_inv_shaped  # shape: same as M

        # rhs = U^T @ (D_inv_M) along last dim -> resulting shape (..., r)
        # FIX: "dr" -> "rd" to match U.t() shape of (r, d)
        rhs = torch.einsum("rd,...d->...r", U.t(), D_inv_M)

        # cholesky_solve: need rhs[..., r, 1]
        rhs_unsq = rhs.unsqueeze(-1)  # (..., r, 1)
        # Use torch.cholesky_solve (works with lower-tri L)
        inner_solved = torch.cholesky_solve(rhs_unsq, L_inner)  # (..., r, 1)
        inner_solved = inner_solved.squeeze(-1)  # (..., r)

        # correction = (U * D_inv[:,None]) @ inner_solved  -> (..., D)
        correction = torch.einsum("dr,...r->...d", U * D_inv.view(-1, 1), inner_solved)

        return D_inv_M - correction

    def log_prob(self, value):
        """
        value: (B, C * S)
        returns: (B,) log probability
        """
        if value.dim() != 2:
            raise ValueError("value must be (B, C*S)")

        B = value.shape[0]
        diff = value - self.loc.unsqueeze(0)          # (B, C*S)
        Z = diff.view(B, self.C, self.D_sp)          # (B, C, S)

        # 1) Apply Sigma_ch^{-1} on the channel axis:
        # Permute so last dim = C for _apply_woodbury_inverse => shape (B, S, C)
        Z_ch_last = Z.permute(0, 2, 1)
        T1_ch_last = self._apply_woodbury_inverse(Z_ch_last, self.ch_woodbury_cache)
        T1 = T1_ch_last.permute(0, 2, 1)              # (B, C, S) == Sigma_ch^{-1} @ Z

        # 2) Apply Sigma_sp^{-1} on the spatial axis (last dim = S)
        # _apply_woodbury_inverse expects last dim = D_sp, so pass T1 directly
        T = self._apply_woodbury_inverse(T1, self.sp_woodbury_cache)  # (B, C, S)

        # 3) Mahalanobis distance = sum over channel & spatial dims
        mahalanobis_dist = torch.sum(Z * T, dim=[1, 2])  # (B,)

        const_term = (self.D_ch * self.D_sp) * torch.log(
            2.0 * torch.tensor(torch.pi, device=value.device, dtype=value.dtype)
        )
        log_p = -0.5 * (const_term + self.log_det_total + mahalanobis_dist)
        return log_p

    def rsample(self, sample_shape=torch.Size()):
        """
        Fully optimized low-rank sampler (vectorized).
        Steps:
          - For spatial side (S large): sample Y_sp columns independently using low-rank sampler:
                for each column j: y_sp_j = D_sp^{1/2} * eps1 + U_sp @ eps2
            This gives Y (S x C) with independent columns sampled from N(0, Sigma_sp).
          - Compute small dense cholesky of Sigma_ch (C x C) and apply to columns:
                Z = (Y @ L_ch.T).T  -> shape (C, S)
          - Flatten and add loc.
        Returns: tensor shaped (N, C*S) where N = prod(sample_shape)
        """
        # shape handling
        shape = self._extended_shape(sample_shape)
        # shape is (..., event_size) in torch Distribution utils; we only care about number of samples
        # interpret shape as (N, )
        if len(shape) == 0:
            N = 1
        else:
            N = shape[0]

        device = self.loc.device
        dtype = self.loc.dtype
        B = N

        S = self.D_sp
        C = self.C

        # Spatial sampling: vectorized across (B, C) columns
        r_sp = self.sp_cov_factor.shape[1]
        # eps1_sp: (B, S, C) for diagonal part
        eps1_sp = _standard_normal((B, S, C), dtype=dtype, device=device)
        # eps2_sp: (B, r_sp, C) for low-rank part (per-column independent)
        eps2_sp = _standard_normal((B, r_sp, C), dtype=dtype, device=device)

        # part1 = D_sp^{1/2} * eps1_sp  -> broadcast with shape (1, S, 1)
        D_sp_sqrt = torch.sqrt(self.sp_cov_diag).view(1, S, 1)
        part1 = D_sp_sqrt * eps1_sp  # (B, S, C)

        # part2 = U_sp @ eps2_sp  -> U_sp: (S, r_sp), eps2_sp: (B, r_sp, C) -> result (B, S, C)
        part2 = torch.einsum("sr,brc->bsc", self.sp_cov_factor, eps2_sp)

        Y = part1 + part2  # (B, S, C) : each slice [:, :, j] is one column sample of spatial side

        # Now apply small dense channel sqrt: compute Sigma_ch dense (C x C), cholesky it
        # Sigma_ch = diag(ch_cov_diag) + U_ch @ U_ch.T
        Sigma_ch = torch.diag(self.ch_cov_diag) + (self.ch_cov_factor @ self.ch_cov_factor.t())
        # Add jitter for numerical stability
        Sigma_ch = Sigma_ch + self.jitter * torch.eye(C, device=device, dtype=dtype)
        L_ch = torch.linalg.cholesky(Sigma_ch)  # (C, C) lower triangular such that L_ch @ L_ch.T = Sigma_ch

        # Multiply: for each batch b, W_b = Y_b @ L_ch.T -> (S, C)
        # Vectorized: W = einsum('bsc,cd->bsd', Y, L_ch.T)
        W = torch.einsum("bsc,cd->bsd", Y, L_ch.t())  # (B, S, C)

        # Convert to Z of shape (B, C, S)
        Z = W.permute(0, 2, 1)  # (B, C, S)

        sample = Z.reshape(B, C * S) + self.loc.unsqueeze(0)  # (B, C*S)
        return sample

    @property
    def loc(self):
        return self._loc
    

if __name__ == "__main__":
    print("--- Running Test Suite for KroneckerProductMVN ---")
    
    # 1. Setup test parameters
    C, H, W = 4, 2, 2
    D_ch, D_sp = C, H * W
    r_ch, r_sp = 2, 2
    B = 8 # Batch size for testing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 2. Generate random parameters for the distributions
    loc = torch.randn(D_ch * D_sp, device=device)
    
    # Channel covariance components
    ch_cov_diag = torch.rand(D_ch, device=device) + 0.5
    ch_cov_factor = torch.randn(D_ch, r_ch, device=device)
    
    # Spatial covariance components
    sp_cov_diag = torch.rand(D_sp, device=device) + 0.5
    sp_cov_factor = torch.randn(D_sp, r_sp, device=device)

    # 3. Build the Ground Truth Distribution
    print("\nStep 1: Constructing ground truth distribution...")
    
    # Construct full dense matrices from low-rank components
    Sigma_ch = torch.diag(ch_cov_diag) + ch_cov_factor @ ch_cov_factor.t()
    Sigma_sp = torch.diag(sp_cov_diag) + sp_cov_factor @ sp_cov_factor.t()
    
    # Construct the full (and very large) Kronecker product covariance
    # This is computationally expensive and exactly what our class avoids.
    print(f"Materializing full Kronecker product matrix of size ({D_ch*D_sp}, {D_ch*D_sp})...")
    Sigma_total = torch.kron(Sigma_ch, Sigma_sp)
    
    # Add jitter for numerical stability, as torch.kron can lose precision
    Sigma_total += 1e-6 * torch.eye(D_ch * D_sp, device=device)

    ground_truth_dist = torch.distributions.MultivariateNormal(
        loc=loc,
        covariance_matrix=Sigma_total
    )
    print("Ground truth distribution created.")

    # 4. Build Our KroneckerProductMVN Distribution
    print("\nStep 2: Constructing optimized KroneckerProductMVN distribution...")
    our_dist = KroneckerProductMVN(
        loc=loc,
        ch_cov=(ch_cov_factor, ch_cov_diag),
        sp_cov=(sp_cov_factor, sp_cov_diag),
        C=C, H=H, W=W
    )
    print("Optimized distribution created.")

    # 5. Generate test data
    test_data = torch.randn(B, D_ch * D_sp, device=device)

    # 6. Compare log_prob results
    print("\nStep 3: Comparing log_prob results...")
    
    # Calculate log_prob using the ground truth
    lp_ground_truth = ground_truth_dist.log_prob(test_data)
    
    # Calculate log_prob using our optimized implementation
    lp_ours = our_dist.log_prob(test_data)
    
    # --- Verification ---
    print(f"\nGround Truth log_prob:\n{lp_ground_truth}")
    print(f"\nOur Optimized log_prob:\n{lp_ours}")
    
    error = torch.abs(lp_ground_truth - lp_ours).mean()
    print(f"\nMean Absolute Error: {error.item():.6f}")

    try:
        torch.testing.assert_close(lp_ground_truth, lp_ours, rtol=1e-3, atol=1e-3)
        print("\n[SUCCESS] log_prob results match ground truth.")
    except AssertionError as e:
        print("\n[FAILURE] log_prob results DO NOT match ground truth.")
        print(e)

    # 7. Compare log determinant
    print("\nStep 4: Comparing log determinant...")
    _, log_det_gt = torch.linalg.slogdet(ground_truth_dist.covariance_matrix)
    log_det_ours = our_dist.log_det_total

    print(f"Ground Truth log_det: {log_det_gt:.4f}")
    print(f"Our Optimized log_det: {log_det_ours:.4f}")

    try:
        torch.testing.assert_close(log_det_gt, log_det_ours, rtol=1e-3, atol=1e-3)
        print("\n[SUCCESS] Log determinant matches ground truth.")
    except AssertionError as e:
        print("\n[FAILURE] Log determinant DO NOT match ground truth.")
        print(e)