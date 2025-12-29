import hydra
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_dataset
from src.models.priors import ClassConditionalPrior, KPMVNPrior, KPMVTPrior, ConditionalKPMVNPrior, ConditionalKPMVTPrior, LowRankMVNPrior
from src.utils.losses import compute_level_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dense_covariance(U, D):
    """Reconstruct dense covariance matrix Sigma = U @ U.T + diag(D)"""
    # U: (..., Dim, Rank)
    # D: (..., Dim)
    # Returns: (..., Dim, Dim)
    low_rank = torch.matmul(U, U.transpose(-1, -2))
    diag = torch.diag_embed(D)
    return low_rank + diag

def compute_condition_number(U, D):
    """Compute condition number of Sigma = U @ U.T + diag(D)"""
    Sigma = get_dense_covariance(U, D)
    # Use torch.linalg.eigvalsh for symmetric matrices
    eigvals = torch.linalg.eigvalsh(Sigma)
    # Avoid division by zero
    min_eig = torch.clamp(eigvals[..., 0], min=1e-10)
    max_eig = eigvals[..., -1]
    return max_eig / min_eig

def compute_kpmvn_stats(prior):
    """Compute stats for KPMVN/MVT priors"""
    # Extract parameters
    if isinstance(prior, (KPMVNPrior, KPMVTPrior)):
        U_ch = prior.cov_ch_factor
        D_ch = torch.exp(prior.log_cov_ch_diag) + prior.eps
        U_sp = prior.cov_sp_factor
        D_sp = torch.exp(prior.log_cov_sp_diag) + prior.eps
        
        cond_ch = compute_condition_number(U_ch, D_ch).item()
        cond_sp = compute_condition_number(U_sp, D_sp).item()
        
        # Total condition number is product of Kronecker factors
        cond_total = cond_ch * cond_sp
        
        log_det = prior._log_det.item()
        
        return {
            "cond_ch": cond_ch,
            "cond_sp": cond_sp,
            "cond_total": cond_total,
            "log_det": log_det
        }
    return {}

def compute_separation(prior):
    """Compute pairwise separation for ClassConditionalPrior"""
    if not isinstance(prior, ClassConditionalPrior):
        return None
        
    K = prior.K
    means = []
    covs = []
    
    # Collect means and covariances
    for p in prior.priors:
        if isinstance(p, (KPMVNPrior, KPMVTPrior)):
            # Mean
            means.append(p._loc.view(-1))
            
            # Covariance (densified)
            # Warning: This can be huge. Check dimensions.
            D_total = p.D_total
            if D_total > 2048:
                return f"Skipped (Dim {D_total} > 2048)"
                
            U_ch = p.cov_ch_factor
            D_ch = torch.exp(p.log_cov_ch_diag) + p.eps
            Sigma_ch = get_dense_covariance(U_ch, D_ch)
            
            U_sp = p.cov_sp_factor
            D_sp = torch.exp(p.log_cov_sp_diag) + p.eps
            Sigma_sp = get_dense_covariance(U_sp, D_sp)
            
            # Kronecker product
            Sigma = torch.kron(Sigma_ch, Sigma_sp)
            
            # Scale for Student-T
            if isinstance(p, KPMVTPrior):
                df = torch.exp(p.log_df) + 2.0
                scale = df / (df - 2.0)
                Sigma = Sigma * scale
                
            covs.append(Sigma)
    
    if not means or not covs:
        return None
        
    means = torch.stack(means) # (K, D)
    covs = torch.stack(covs)   # (K, D, D)
    
    # Compute pairwise distances
    # d_ij^2 = (mu_i - mu_j)^T @ ((Sigma_i + Sigma_j)/2)^-1 @ (mu_i - mu_j)
    distances = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = means[i] - means[j]
            avg_cov = 0.5 * (covs[i] + covs[j])
            
            # Solve linear system instead of explicit inverse for stability
            # avg_cov @ x = diff => x = avg_cov^-1 @ diff
            # dist = diff^T @ x
            try:
                # Use Cholesky solve if positive definite
                L = torch.linalg.cholesky(avg_cov)
                x = torch.cholesky_solve(diff.unsqueeze(1), L).squeeze(1)
                dist = torch.dot(diff, x)
                distances.append(dist.item())
            except RuntimeError:
                # Fallback to solve (slower, maybe less stable)
                x = torch.linalg.solve(avg_cov, diff)
                dist = torch.dot(diff, x)
                distances.append(dist.item())
                
    if not distances:
        return 0.0
        
    return np.mean(distances)

def collect_posterior_stats(model, loader, device, num_batches=10):
    """Collect empirical posterior statistics (z) from the model"""
    model.eval()
    
    z_stats = {} # level -> {class_idx -> list of z}
    
    print(f"\nCollecting posterior stats over {num_batches} batches...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            
            x, y = x.to(device), y.to(device)
            outs, _ = model(x)
            
            for level_idx, (h, z) in enumerate(outs):
                if level_idx not in z_stats:
                    z_stats[level_idx] = {}
                
                # Flatten z: (B, C, H, W) -> (B, D)
                z_flat = z.view(z.shape[0], -1)
                
                for b in range(x.shape[0]):
                    label = y[b].item()
                    if label not in z_stats[level_idx]:
                        z_stats[level_idx][label] = []
                    z_stats[level_idx][label].append(z_flat[b].cpu())
    
    # Aggregate
    results = {}
    for level_idx, class_data in z_stats.items():
        results[level_idx] = {}
        for label, z_list in class_data.items():
            if len(z_list) < 2:
                continue
            z_tensor = torch.stack(z_list).to(device) # (N, D)
            mean = torch.mean(z_tensor, dim=0)
            # Compute covariance if N is large enough, else diagonal
            # For high dim, full cov is expensive. Let's stick to diagonal for KL approx?
            # Or just store the samples and compute KL later if needed.
            # Let's compute full mean and diagonal variance for now to save memory
            var = torch.var(z_tensor, dim=0)
            results[level_idx][label] = {"mean": mean, "var": var, "n": len(z_list)}
            
    return results

def compute_kl_divergence(emp_mean, emp_var, prior):
    """
    Compute KL(q || p) where q is empirical Gaussian (diag cov) and p is Prior.
    KL(N0 || N1) = 0.5 * [ tr(Sigma1^-1 Sigma0) + (mu1 - mu0)^T Sigma1^-1 (mu1 - mu0) - k + ln(|Sigma1|/|Sigma0|) ]
    
    Here q = N(emp_mean, diag(emp_var))
    p = Prior (KPMVN)
    
    We need efficient computation using Kronecker structure of p.
    Sigma_p = Sigma_ch ⊗ Sigma_sp
    Sigma_p^-1 = Sigma_ch^-1 ⊗ Sigma_sp^-1
    """
    if not isinstance(prior, (KPMVNPrior, KPMVTPrior)):
        return float('nan')
        
    # 1. Reconstruct Prior parameters
    U_ch = prior.cov_ch_factor
    D_ch = torch.exp(prior.log_cov_ch_diag) + prior.eps
    Sigma_ch = get_dense_covariance(U_ch, D_ch)
    
    U_sp = prior.cov_sp_factor
    D_sp = torch.exp(prior.log_cov_sp_diag) + prior.eps
    Sigma_sp = get_dense_covariance(U_sp, D_sp)
    
    # Invert Prior components
    # Use cholesky inverse for stability
    try:
        L_ch = torch.linalg.cholesky(Sigma_ch)
        Sigma_ch_inv = torch.cholesky_inverse(L_ch)
        L_sp = torch.linalg.cholesky(Sigma_sp)
        Sigma_sp_inv = torch.cholesky_inverse(L_sp)
    except RuntimeError:
        Sigma_ch_inv = torch.linalg.inv(Sigma_ch)
        Sigma_sp_inv = torch.linalg.inv(Sigma_sp)
        
    # Prior LogDet
    log_det_p = prior._log_det
    
    # Posterior (Empirical) LogDet
    # q is diagonal: sum(log(var))
    log_det_q = torch.sum(torch.log(emp_var + 1e-8))
    
    k = emp_mean.shape[0]
    
    # Term 1: tr(Sigma_p^-1 Sigma_q)
    # Sigma_q is diagonal. tr(A D) = sum(diag(A) * diag(D))
    # We need diagonal of Sigma_p^-1 = Sigma_ch^-1 ⊗ Sigma_sp^-1
    # diag(A ⊗ B) = diag(A) ⊗ diag(B)
    diag_p_inv = torch.kron(torch.diagonal(Sigma_ch_inv), torch.diagonal(Sigma_sp_inv))
    trace_term = torch.sum(diag_p_inv * emp_var)
    
    # Term 2: (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q)
    diff = prior._loc.view(-1) - emp_mean
    # diff^T (Sigma_ch^-1 ⊗ Sigma_sp^-1) diff
    # Reshape diff to (C, S)
    diff_mat = diff.view(prior.C, prior.D_sp) # (C, S)
    # (A ⊗ B) vec(X) = vec(B X A^T)
    # Here we want vec(X)^T (A ⊗ B) vec(X) = tr(X^T B X A^T) ? No.
    # Let's use the property: (A ⊗ B) = (A ⊗ I) (I ⊗ B)
    # Or just compute explicitly if dims are small.
    # Actually, vec(X)^T (A ⊗ B) vec(X) = tr( A X B^T X^T ) ?
    # Let's verify: (A ⊗ B) vec(X) = vec(B X A^T).
    # So vec(X)^T vec(B X A^T) = tr(X^T B X A^T).
    # Here A = Sigma_ch^-1, B = Sigma_sp^-1.
    # So term2 = tr( diff_mat^T @ Sigma_sp_inv @ diff_mat @ Sigma_ch_inv.T )
    # Since covariances are symmetric: tr( diff_mat^T @ Sigma_sp_inv @ diff_mat @ Sigma_ch_inv )
    
    term2 = torch.trace(diff_mat.t() @ Sigma_sp_inv @ diff_mat @ Sigma_ch_inv)
    
    kl = 0.5 * (trace_term + term2 - k + log_det_p - log_det_q)
    return kl.item()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def eval_priors(cfg: DictConfig):
    print(f"Device: {device}")
    
    # Load data
    train_loader, test_loader = load_dataset(cfg.data)
    input_shape = next(iter(train_loader))[0].shape[1:]  # (C, H, W)
    K = cfg.data.dataset.num_classes
    
    if cfg.model._target_ == "src.models.glow.DGLOWNetwork":
        from src.models.glow import DGLOWNetwork
        output_shapes = DGLOWNetwork.output_shapes(input_shape, cfg.model.num_levels)
    else:
        raise NotImplementedError(f"Model {cfg.model._target_} not supported.")
    
    # Initialize model (includes priors)
    print("\nInitializing Model...")
    # We need to manually construct std_per_level placeholder as in train.py
    # But actually, we can just instantiate the model and load weights.
    # The model constructor needs std_per_level, but if we load weights, it might be overwritten?
    # Wait, std_per_level is used for initialization of ActNorm.
    # Let's just pass zeros, it will be loaded from checkpoint.
    std_per_level = [torch.zeros(s[0], device=device) for s in output_shapes]
    
    model = hydra.utils.instantiate(
        cfg.model,
        input_shape=input_shape, 
        std_per_level=std_per_level,
        _convert_="partial"
    ).to(device)
    
    # Reconstruct priors list for easy access (similar to train.py)
    level_priors = []
    level_priors_params = nn.ModuleList()
    
    # We need to instantiate priors separately to get the objects, 
    # OR we can try to extract them if they were part of the model?
    # No, in train.py, priors are separate from the model.
    # So we must instantiate them exactly as in train.py.
    
    print("Initializing Priors...")
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors.values()):
            C, H, W = output_shapes[i]
            split = prior_cfg.split
            noise_count, struct_count, sem_count = split
            
            noise_prior, struct_prior, sem_prior = None, None, None
            level_params = nn.ModuleList()
            
            if noise_count > 0:
                theta_list = hydra.utils.instantiate(
                    prior_cfg.zero_init, 
                    K=1,
                    C=noise_count, H=H, W=W,
                    rank=prior_cfg.rank
                )
                noise_prior = hydra.utils.instantiate(
                    prior_cfg.cls,
                    **theta_list[0]
                ).to(device)
                level_params.append(noise_prior)
            
            if struct_count > 0:
                if prior_cfg.conditional_cls._target_ == "src.models.priors.ConditionalMixturePrior":
                    num_components = prior_cfg.conditional_cls.num_components
                    cond_cls_cfg = prior_cfg.conditional_cls.copy()
                    del cond_cls_cfg.num_components
                    theta_list = hydra.utils.instantiate(
                        prior_cfg.class_conditional_init,
                        K=num_components,
                        C=struct_count, H=H, W=W,
                        rank=prior_cfg.rank
                    )
                    components = [
                        hydra.utils.instantiate(prior_cfg.cls, **theta) for theta in theta_list
                    ]
                    struct_prior = hydra.utils.instantiate(
                        cond_cls_cfg,
                        components=components,
                        h_channels=C
                    ).to(device)
                    level_params.append(struct_prior)
                elif prior_cfg.conditional_cls._target_ in [
                    "src.models.priors.ConditionalKPMVNPrior",
                    "src.models.priors.ConditionalKPMVTPrior"
                ]:
                    struct_prior = hydra.utils.instantiate(
                        prior_cfg.conditional_cls,
                        h_channels=struct_count,
                        z_channels=C,
                        H=H, W=W,
                        rank=prior_cfg.rank
                    ).to(device)
                    level_params.append(struct_prior)
            
            if sem_count > 0:
                theta_list = hydra.utils.instantiate(
                    prior_cfg.class_conditional_init,
                    K=K,
                    C=sem_count, H=H, W=W,
                    rank=prior_cfg.rank
                )
                sem_prior = ClassConditionalPrior([
                    hydra.utils.instantiate(prior_cfg.cls, **theta) for theta in theta_list
                ]).to(device)
                level_params.append(sem_prior)
                
            level_priors.append([noise_prior, struct_prior, sem_prior])
            level_priors_params.append(level_params)

    # Load Checkpoint
    if cfg.training.ckpt is not None:
        print(f"\nLoading checkpoint from {cfg.training.ckpt}")
        checkpoint = torch.load(cfg.training.ckpt, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'prior_state_dict' in checkpoint:
            level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
        print("Checkpoint loaded.")
    else:
        print("Warning: No checkpoint specified. Using random initialization.")

    # 1. Prior Diagnostics
    print("\n" + "="*50)
    print("PRIOR DIAGNOSTICS")
    print("="*50)
    
    for i, (noise, struct, sem) in enumerate(level_priors):
        print(f"\nLevel {i}:")
        
        if noise:
            stats = compute_kpmvn_stats(noise)
            print(f"  [Noise] {noise}")
            print(f"    Cond(Ch): {stats.get('cond_ch', 0):.2e}, Cond(Sp): {stats.get('cond_sp', 0):.2e}, Total: {stats.get('cond_total', 0):.2e}")
            print(f"    LogDet: {stats.get('log_det', 0):.2f}")
            
        if struct:
            print(f"  [Struct] {struct}")
            # Struct is conditional, so it's harder to diagnose static covariance.
            # We can check the output distribution for a zero input or random input?
            # Or just inspect the heads.
            pass
            
        if sem:
            print(f"  [Semantic] {sem}")
            sep = compute_separation(sem)
            if sep is not None:
                print(f"    Avg Separation Distance: {sep:.4f}")
            
            # Stats for first component
            if hasattr(sem, "priors") and len(sem.priors) > 0:
                stats = compute_kpmvn_stats(sem.priors[0])
                print(f"    Comp 0 - Cond: {stats.get('cond_total', 0):.2e}, LogDet: {stats.get('log_det', 0):.2f}")

    # 2. Posterior Alignment
    print("\n" + "="*50)
    print("POSTERIOR ALIGNMENT (KL Divergence)")
    print("="*50)
    
    # Collect posterior stats
    post_stats = collect_posterior_stats(model, test_loader, device, num_batches=20)
    
    for i, (noise, struct, sem) in enumerate(level_priors):
        print(f"\nLevel {i}:")
        
        # We need to know which part of z corresponds to noise/struct/sem
        # The split is defined in cfg.
        split = cfg.level_priors.priors[f"l{i}"].split # e.g. [0, 6, 0]
        n_c, s_c, sem_c = split
        
        # Get stats for this level
        if i not in post_stats:
            continue
            
        level_data = post_stats[i]
        
        # Average KL across classes
        kl_sem_sum = 0.0
        count = 0
        
        for label, stats in level_data.items():
            mean = stats["mean"] # (D,)
            var = stats["var"]   # (D,)
            
            # Split mean/var into noise/struct/sem parts
            # The order in z depends on how they are concatenated.
            # In train.py: compute_level_logits concatenates [z_noise, z_struct, z_sem] ?
            # No, compute_level_logits takes z and splits it.
            # z is split using torch.split(z, split, dim=1)
            
            # We need to split the flattened mean/var.
            # But wait, z is (C, H, W). Flattened is (C*H*W).
            # The split is along C.
            # So we need to reshape, split, and flatten again.
            
            C, H, W = output_shapes[i]
            mean_reshaped = mean.view(C, H, W)
            var_reshaped = var.view(C, H, W)
            
            m_splits = torch.split(mean_reshaped, split, dim=0)
            v_splits = torch.split(var_reshaped, split, dim=0)
            
            m_noise, m_struct, m_sem = m_splits
            v_noise, v_struct, v_sem = v_splits
            
            # Semantic KL
            if sem and sem_c > 0:
                # Get prior for this class
                prior_k = sem.priors[label]
                
                # Flatten
                m_sem_flat = m_sem.reshape(-1)
                v_sem_flat = v_sem.reshape(-1)
                
                kl = compute_kl_divergence(m_sem_flat, v_sem_flat, prior_k)
                kl_sem_sum += kl
                count += 1
        
        if sem and count > 0:
            print(f"  [Semantic] Avg KL(q || p): {kl_sem_sum / count:.4f}")

if __name__ == "__main__":
    eval_priors()
