import torch
from torch.distributions import Distribution
from torch.distributions.utils import _standard_normal

class KroneckerProductMVN(Distribution):
    """
    Multivariate Normal with covariance Sigma = Sigma_ch ⊗ Sigma_sp.
    Supports arbitrary batch shapes for loc, ch_cov, and sp_cov.
    """
    arg_constraints = {}    # type: ignore
    has_rsample = True

    def __init__(self, loc, ch_cov, sp_cov, C, H, W, jitter=1e-4):
        """
        loc: (..., C * S) mean vector
        ch_cov: tuple (U_ch, D_ch) where U_ch: (..., C, r_ch), D_ch: (..., C)
        sp_cov: tuple (U_sp, D_sp) where U_sp: (..., S, r_sp), D_sp: (..., S)
        """
        self.C, self.H, self.W = C, H, W
        self.D_ch = C
        self.D_sp = H * W
        S = self.D_sp

        # 1. Broadcast all parameters to a common batch shape
        # We need to look at the batch dimensions of loc, ch params, and sp params
        # loc: (Batch..., CS)
        # ch_d: (Batch..., C)
        # sp_d: (Batch..., S)
        # We broadcast the *leading* dimensions.
        batch_shape = torch.broadcast_shapes(
            loc.shape[:-1],
            ch_cov[1].shape[:-1], # ch_cov_diag
            sp_cov[1].shape[:-1]  # sp_cov_diag
        )
        
        # Reshape loc to (Batch..., C, H, W) for event_shape alignment
        self._loc = (
            loc
            .expand(batch_shape + (C * S,))
            .clone()
            .reshape(batch_shape + (C, H, W))
        )
        
        # Broadcast Covariance parameters
        # Factor U: (..., D, r) -> expand to (batch..., D, r)
        # Diag D:   (..., D)    -> expand to (batch..., D)
        self.ch_cov_factor = ch_cov[0].expand(batch_shape + (C, -1)).clone()
        self.ch_cov_diag   = ch_cov[1].expand(batch_shape + (C,)).clone()
        self.sp_cov_factor = sp_cov[0].expand(batch_shape + (S, -1)).clone()
        self.sp_cov_diag   = sp_cov[1].expand(batch_shape + (S,)).clone()
        
        self.jitter = jitter

        # 2. Precompute log-determinants (batched)
        self._log_det_ch = self._compute_log_det(self.ch_cov_diag, self.ch_cov_factor) # (Batch...)
        self._log_det_sp = self._compute_log_det(self.sp_cov_diag, self.sp_cov_factor) # (Batch...)
        self.log_det_total = self.D_sp * self._log_det_ch + self.D_ch * self._log_det_sp

        # 3. Precompute Woodbury caches (batched)
        self.ch_woodbury_cache = self._precompute_woodbury(self.ch_cov_diag, self.ch_cov_factor, jitter)
        self.sp_woodbury_cache = self._precompute_woodbury(self.sp_cov_diag, self.sp_cov_factor, jitter)

        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([C, H, W]))

    @staticmethod
    def _compute_log_det(cov_diag: torch.Tensor, cov_factor: torch.Tensor) -> torch.Tensor:
        """
        cov_diag: (..., D)
        cov_factor: (..., D, r)
        """
        # 1. Clamp D to ensure numerical stability
        cov_diag = torch.clamp(cov_diag, min=1e-6)
        
        r = cov_factor.shape[-1]
        D_inv = 1.0 / cov_diag
        # U_scaled = U * D^-1. Broadcasting (..., D, r) * (..., D, 1)
        U_scaled = cov_factor * D_inv.unsqueeze(-1) 
        
        # M = I + U^T D^-1 U
        # matmul: (..., r, D) @ (..., D, r) -> (..., r, r)
        M = torch.eye(r, device=cov_diag.device, dtype=cov_diag.dtype) + torch.matmul(cov_factor.transpose(-1, -2), U_scaled)
        
        # 2. Symmetrize M to correct floating point drift
        M = 0.5 * (M + M.transpose(-1, -2))
        
        sign, logabsdet_M = torch.linalg.slogdet(M)
        logdet_D = torch.sum(torch.log(cov_diag), dim=-1)
        return logdet_D + logabsdet_M

    def _precompute_woodbury(self, D, U, jitter):
        """
        Precomputes Woodbury parts. Handles batching natively.
        D: (..., D_dim)
        U: (..., D_dim, r)
        """
        # 1. Clamp D to ensure numerical stability
        D = torch.clamp(D, min=1e-6)
        
        r = U.shape[-1]
        D_inv = 1.0 / D
        
        # M_inner = I + U^T D^-1 U
        # U * D_inv.unsqueeze(-1) -> (..., D, r)
        # U.t() handles batching if we use transpose(-1, -2)
        U_scaled = U * D_inv.unsqueeze(-1)
        M_inner = torch.eye(r, device=D.device, dtype=D.dtype) + torch.matmul(U.transpose(-1, -2), U_scaled)
        
        # 2. Symmetrize M_inner to correct floating point drift
        M_inner = 0.5 * (M_inner + M_inner.transpose(-1, -2))
        
        # 3. Robust Jitter: Scale jitter by the diagonal magnitude
        # This ensures jitter is effective even if matrix elements are large
        diag_mean = torch.diagonal(M_inner, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        scaled_jitter = jitter * torch.maximum(torch.ones_like(diag_mean), diag_mean)
        
        M_inner = M_inner + scaled_jitter * torch.eye(r, device=D.device, dtype=D.dtype)
        
        # Cholesky decomposition of the r x r matrix
        L_inner = torch.linalg.cholesky(M_inner) 
        
        return {"D_inv": D_inv, "L_inner": L_inner, "U": U}

    def _unsqueeze_cache(self, cache, dim_idx):
        """
        Helper to insert a singleton dimension into cached tensors to support 
        broadcasting against 'interjected' dimensions (like S in the channel step).
        
        If dim_idx is -2 (insert before the last dim):
           D_inv (..., D)    -> (..., 1, D)
           U     (..., D, r) -> (..., 1, D, r)
           L     (..., r, r) -> (..., 1, r, r)
        """
        new_cache = {}
        for k, v in cache.items():
            # If v is a matrix (U, L), we need to insert at dim_idx - 1 (because it has an extra dim at the end)
            # If v is a vector (D_inv), we insert at dim_idx
            
            # Logic:
            # We want to insert a '1' such that it aligns with the 'interjected' dimension in M.
            # M shape: (..., Interjected, D)
            # D_inv shape: (..., D). We want (..., 1, D)
            # U shape: (..., D, r). We want (..., 1, D, r)
            # L shape: (..., r, r). We want (..., 1, r, r)
            
            # So effectively we insert at -2 for D_inv, and -3 for U and L.
            
            if k == "D_inv":
                new_cache[k] = v.unsqueeze(-2)
            else:
                new_cache[k] = v.unsqueeze(-3)
        return new_cache

    def _apply_woodbury_inverse(self, M, cache):
        """
        M: (..., D)  (Last dim matches covariance dim)
        cache: Batched parameters aligned with M's leading dims.
        """
        D_inv = cache["D_inv"]      # (..., D)
        L_inner = cache["L_inner"]  # (..., r, r)
        U = cache["U"]              # (..., D, r)

        # 1. D^-1 M
        # D_inv is (..., D). M is (..., D). 
        # Standard broadcasting works if batch dims match.
        D_inv_M = M * D_inv 

        # 2. rhs = U^T @ (D^-1 M)
        # U: (..., D, r). D_inv_M: (..., D)
        # We want contraction over D. Result (..., r)
        # einsum '...dr, ...d -> ...r' robustly handles batch dimensions
        rhs = torch.einsum("...dr,...d->...r", U, D_inv_M)

        # 3. Solve (I + U^T D^-1 U) X = rhs
        # rhs: (..., r). unsqueeze to (..., r, 1) for cholesky_solve
        inner_solved = torch.cholesky_solve(rhs.unsqueeze(-1), L_inner).squeeze(-1) # (..., r)

        # 4. correction = D^-1 U @ inner_solved
        # term: (U * D_inv_expanded) -> (..., D, r)
        # inner_solved: (..., r)
        # result: (..., D)
        U_Dinv = U * D_inv.unsqueeze(-1)
        correction = torch.einsum("...dr,...r->...d", U_Dinv, inner_solved)

        return D_inv_M - correction

    def log_prob(self, value):
        """
        value: (Sample_Batch..., Batch..., C, H, W)
        """
        # Ensure value aligns with loc
        diff = value - self.loc # (..., C, H, W)
        
        # Reshape to (..., C, S)
        # We use shape[:-3] to preserve all leading sample/batch dimensions
        Z = diff.reshape(diff.shape[:-3] + (self.C, self.D_sp))

        # --- Step 1: Apply Sigma_ch^-1 ---
        # Z is (..., C, S). We permute to (..., S, C) to put C last.
        Z_perm = Z.transpose(-2, -1) # (..., S, C)
        
        # CRITICAL: Z_perm has shape (..., S, C). 
        # The channel covariance params have shape (Batch..., C).
        # They align with 'C', but 'S' is sitting in between Batch and C.
        # We must insert a singleton dim into the cache params to bridge 'S'.
        ch_cache_expanded = self._unsqueeze_cache(self.ch_woodbury_cache, -2)
        
        T1_perm = self._apply_woodbury_inverse(Z_perm, ch_cache_expanded)
        T1 = T1_perm.transpose(-2, -1) # Swap back -> (..., C, S)
        
        # --- Step 2: Apply Sigma_sp^-1 ---
        # T1 is (..., C, S). S is last.
        # Spatial params are (Batch..., S).
        # 'C' is sitting between Batch and S.
        # We must insert a singleton dim into the cache params to bridge 'C'.
        sp_cache_expanded = self._unsqueeze_cache(self.sp_woodbury_cache, -2)
        
        T = self._apply_woodbury_inverse(T1, sp_cache_expanded) # (..., C, S)

        # --- Step 3: Mahalanobis ---
        # tr(Z^T T) = sum(Z * T)
        mahalanobis_dist = torch.sum(Z * T, dim=[-2, -1])

        const_term = (self.D_ch * self.D_sp) * torch.log(
            2.0 * torch.tensor(torch.pi, device=value.device, dtype=value.dtype)
        )
        log_p = -0.5 * (const_term + self.log_det_total + mahalanobis_dist)
        return log_p

    def anisotropy_loss(self) -> torch.Tensor:
        """
        Computes the KL divergence between this distribution and a volume-matched 
        isotropic Gaussian. This serves as an anisotropy penalty.
        
        Equivalent to minimizing the log-ratio of the Arithmetic Mean to the 
        Geometric Mean of the eigenvalues (AM/GM inequality).
        
        Returns:
            loss: (Batch...,) The anisotropy loss for each batch element.
        """
        # 1. Channel Component Anisotropy
        # tr(Sigma_ch) = sum(D_ch) + ||U_ch||_F^2
        # ch_cov_diag: (Batch..., C)
        # ch_cov_factor: (Batch..., C, r)
        tr_ch = torch.sum(self.ch_cov_diag, dim=-1) + torch.sum(self.ch_cov_factor.pow(2), dim=[-2, -1])
        d_ch = float(self.D_ch)
        
        # log(AM/GM) = log(tr/d) - log_det/d
        #            = log(tr) - log(d) - log_det/d
        log_am_gm_ch = torch.log(tr_ch + self.jitter) - torch.log(torch.tensor(d_ch, device=tr_ch.device)) - (self._log_det_ch / d_ch)
        
        # 2. Spatial Component Anisotropy
        # sp_cov_diag: (Batch..., S)
        # sp_cov_factor: (Batch..., S, r)
        tr_sp = torch.sum(self.sp_cov_diag, dim=-1) + torch.sum(self.sp_cov_factor.pow(2), dim=[-2, -1])
        d_sp = float(self.D_sp)
        
        log_am_gm_sp = torch.log(tr_sp + self.jitter) - torch.log(torch.tensor(d_sp, device=tr_sp.device)) - (self._log_det_sp / d_sp)
        
        return log_am_gm_ch + log_am_gm_sp

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates samples: (sample_shape + batch_shape + event_shape)
        """
        # 1. Setup Shapes
        batch_shape = self.batch_shape
        shape = sample_shape + batch_shape  # type: ignore
        
        device = self.loc.device
        dtype = self.loc.dtype
        S = self.D_sp
        C = self.C

        # 2. Spatial Sampling (Low Rank + Diagonal)
        # We generate noise for shape: (Sample..., Batch..., S, C)
        # Note: We treat 'C' as independent spatial samples initially.
        
        # eps1_sp: Standard normal noise for the spatial diagonal part
        eps1_sp = _standard_normal(shape + (S, C), dtype=dtype, device=device)
        
        # eps2_sp: Standard normal noise for the spatial low-rank part
        r_sp = self.sp_cov_factor.shape[-1]
        eps2_sp = _standard_normal(shape + (r_sp, C), dtype=dtype, device=device)

        # Apply Diagonal Variance
        # sp_cov_diag: (Batch..., S). 
        # We need to broadcast it against (Sample..., Batch..., S, C)
        # Reshape to (1..., Batch..., S, 1) to align correctly.
        # We calculate the number of sample dims to prepend '1's.
        sample_dims = len(sample_shape)
        D_sp_view = self.sp_cov_diag
        for _ in range(sample_dims):
            D_sp_view = D_sp_view.unsqueeze(0)
        
        # Using unsqueeze is safer than reshape as it doesn't depend on memory layout.
        D_sp_sqrt = torch.sqrt(D_sp_view).unsqueeze(-1) # (1..., Batch..., S, 1)
        part1 = D_sp_sqrt * eps1_sp

        # Apply Low-Rank Variance
        # part2 = U_sp @ eps2_sp
        # U_sp: (Batch..., S, r). eps2_sp: (Sample..., Batch..., r, C)
        # Matmul broadcasts the batch dimensions automatically.
        part2 = torch.matmul(self.sp_cov_factor, eps2_sp)
        
        # Y is the spatially correlated, channel independent intermediate
        Y = part1 + part2 # (Sample..., Batch..., S, C)

        # 3. Channel Correlation (Dense Cholesky)
        # Sigma_ch = D + UU^T
        U_ch = self.ch_cov_factor
        D_ch = torch.diag_embed(self.ch_cov_diag)
        Sigma_ch = D_ch + torch.matmul(U_ch, U_ch.transpose(-1, -2))
        
        # Add jitter for stability
        I_C = torch.eye(C, device=device, dtype=dtype)
        Sigma_ch = Sigma_ch + self.jitter * I_C
        
        # Cholesky decomp -> (Batch..., C, C)
        L_ch = torch.linalg.cholesky(Sigma_ch)

        # We want to correlate the 'C' dimension of Y.
        # Y is (..., S, C). We need Y @ L_ch^T.
        # Matmul broadcasts L_ch (Batch..., C, C) against Y (Sample..., Batch..., S, C)
        W = torch.matmul(Y, L_ch.transpose(-1, -2)) 

        # 4. Final Reshape and Loc
        # Transpose S and C to match flattening order (C outer, S inner) -> (..., C, S)
        # Then reshape to (..., C, H, W)
        Z = W.transpose(-2, -1).reshape(shape + (self.C, self.H, self.W))
        
        return self.loc + Z

    @property
    def loc(self):
        return self._loc
    
    
if __name__ == "__main__":
    print("--- Running Test Suite for KroneckerProductMVN ---")
    
    # 1. Setup test parameters
    C, H, W = 4, 32, 32
    D_ch, D_sp = C, H * W
    r_ch, r_sp = 2, 2
    
    # Use a multi-dimensional batch shape to test broadcasting
    batch_shape = (3, 2) 
    sample_shape = (500,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 2. Generate random BATCHED parameters for the distributions
    print(f"\nStep 1: Generating batched parameters with batch_shape={batch_shape}")
    loc = torch.randn(batch_shape + (D_ch * D_sp,), device=device)
    
    # Channel covariance components
    ch_cov_diag = torch.rand(batch_shape + (D_ch,), device=device) + 0.5
    ch_cov_factor = torch.randn(batch_shape + (D_ch, r_ch), device=device)
    
    # Spatial covariance components
    sp_cov_diag = torch.rand(batch_shape + (D_sp,), device=device) + 0.5
    sp_cov_factor = torch.randn(batch_shape + (D_sp, r_sp), device=device)

    # 3. Build the Ground Truth Distribution (Iteratively)
    print("\nStep 2: Constructing ground truth distribution (iteratively)...")
    
    # Since torch.kron is not batched, we must loop.
    lp_ground_truth_list = []
    log_det_gt_list = []
    
    # Generate test data with sample and batch dimensions
    test_data = torch.randn(sample_shape + batch_shape + (C, H, W), device=device)

    # This nested loop simulates iterating over the batch dimensions
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            # Select the parameters for this specific batch item
            loc_ij = loc[i, j]
            ch_diag_ij = ch_cov_diag[i, j]
            ch_factor_ij = ch_cov_factor[i, j]
            sp_diag_ij = sp_cov_diag[i, j]
            sp_factor_ij = sp_cov_factor[i, j]
            
            # Construct the dense covariance for this single item
            Sigma_ch = torch.diag(ch_diag_ij) + ch_factor_ij @ ch_factor_ij.t()
            Sigma_sp = torch.diag(sp_diag_ij) + sp_factor_ij @ sp_factor_ij.t()
            Sigma_total = torch.kron(Sigma_ch, Sigma_sp)
            Sigma_total += 1e-5 * torch.eye(D_ch * D_sp, device=device) # Increased jitter

            # Create a single MVN for this batch item
            gt_dist_ij = torch.distributions.MultivariateNormal(
                loc=loc_ij,
                covariance_matrix=Sigma_total
            )
            
            # Calculate log_prob for the corresponding slice of test data
            # Flatten for GT
            test_data_ij_flat = test_data[:, i, j, ...].reshape(-1, D_ch * D_sp)
            lp_ground_truth_list.append(gt_dist_ij.log_prob(test_data_ij_flat))
            
            # Store log determinant
            _, log_det_gt_ij = torch.linalg.slogdet(Sigma_total)
            log_det_gt_list.append(log_det_gt_ij)

    # Stack results to match the expected batched output shape
    lp_ground_truth = torch.stack([
        torch.stack(lp_ground_truth_list[i*batch_shape[1]:(i+1)*batch_shape[1]], dim=1) 
        for i in range(batch_shape[0])
    ], dim=1)
    log_det_gt = torch.tensor(log_det_gt_list, device=device).reshape(batch_shape)
    print("Ground truth calculations complete.")


    # 4. Build Our Batched KroneckerProductMVN Distribution
    print("\nStep 3: Constructing single, batched KroneckerProductMVN distribution...")
    our_dist = KroneckerProductMVN(
        loc=loc,
        ch_cov=(ch_cov_factor, ch_cov_diag),
        sp_cov=(sp_cov_factor, sp_cov_diag),
        C=C, H=H, W=W,
        jitter=1e-5 # Match jitter
    )
    print("Optimized batched distribution created.")
    assert our_dist.batch_shape == batch_shape, f"Batch shape mismatch! Expected {batch_shape}, got {our_dist.batch_shape}"
    assert our_dist.event_shape == (C, H, W), f"Event shape mismatch! Expected {(C, H, W)}, got {our_dist.event_shape}"


    # 5. Compare log_prob results
    print("\nStep 4: Comparing log_prob results...")
    
    # Calculate log_prob using our single batched implementation
    lp_ours = our_dist.log_prob(test_data)
    
    # --- Verification ---
    print(f"Ground Truth log_prob shape: {lp_ground_truth.shape}")
    print(f"Our Optimized log_prob shape: {lp_ours.shape}")
    
    assert lp_ours.shape == sample_shape + batch_shape, f"Log_prob shape error! Expected {sample_shape + batch_shape}, got {lp_ours.shape}"

    try:
        torch.testing.assert_close(lp_ground_truth, lp_ours, rtol=1e-3, atol=1e-3)
        print("\n[SUCCESS] Batched log_prob results match ground truth.")
    except AssertionError as e:
        print("\n[FAILURE] Batched log_prob results DO NOT match ground truth.")
        print(e)

    # 6. Compare log determinant
    print("\nStep 5: Comparing log determinant...")
    log_det_ours = our_dist.log_det_total

    print(f"Ground Truth log_det shape: {log_det_gt.shape}")
    print(f"Our Optimized log_det shape: {log_det_ours.shape}")
    
    assert log_det_ours.shape == batch_shape, f"Log_det shape error! Expected {batch_shape}, got {log_det_ours.shape}"

    try:
        torch.testing.assert_close(log_det_gt, log_det_ours, rtol=1e-3, atol=1e-3)
        print("\n[SUCCESS] Batched Log determinant matches ground truth.")
    except AssertionError as e:
        print("\n[FAILURE] Batched Log determinant DO NOT match ground truth.")
        print(e)

    # 7. Test rsample
    print("\nStep 6: Testing rsample()...")
    samples = our_dist.rsample(sample_shape)
    expected_shape = sample_shape + batch_shape + (C, H, W)
    print(f"Expected sample shape: {expected_shape}")
    print(f"Actual sample shape:   {samples.shape}")
    assert samples.shape == expected_shape, "rsample() returned incorrect shape!"
    
    # Check basic statistics
    sample_mean = samples.mean(dim=list(range(len(sample_shape))))
    sample_std = samples.std(dim=list(range(len(sample_shape))))
    
    # The mean of samples should be close to the distribution's loc
    # loc is (Batch..., C, H, W)
    mean_error = torch.abs(sample_mean - our_dist.loc).mean()
    print(f"Mean error between samples and loc: {mean_error:.4f}")
    # This is a weak test, but confirms the loc is being used.
    # A low error suggests correctness. For a large sample_shape, this should be very small.
    assert mean_error < 0.1, "Sample mean deviates significantly from loc."

    print("\n[SUCCESS] rsample() tests passed.")