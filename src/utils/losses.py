import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)


def get_target_distributions(latent_mus, latent_Ls):
    """
    Returns a list of MultivariateNormal distributions, one for each level.
    latent_mus: List of (M_i, D) tensors (Shared Attribute Means)
    latent_Ls: List of (M_i, D, D) tensors (Shared Attribute Covariances)
    """
    dists = []
    for mu, L_param in zip(latent_mus, latent_Ls):
        D = L_param.shape[-1]
        if D == 0:
            dists.append(None)
            continue

        # 1. Construct valid Cholesky factor
        # L_param is unconstrained
        # Diagonal must be positive
        diag = F.softplus(torch.diagonal(L_param, dim1=-2, dim2=-1)) + 1e-5
        L = torch.tril(L_param, diagonal=-1) + torch.diag_embed(diag)
        
        # 2. Enforce Average Unit Determinant (Global Volume Preservation)
        # We want the geometric mean of determinants to be 1.
        # This is equivalent to the arithmetic mean of log-determinants being 0.
        
        # log_det per cluster: (M,)
        log_det_per_cluster = torch.sum(torch.log(diag), dim=-1) 
        
        # Average log_det across clusters: scalar
        avg_log_det = log_det_per_cluster.mean()
        
        # We need to scale L such that the new avg_log_det is 0.
        scale = torch.exp(-avg_log_det / D)
        
        # Apply global scale to all clusters
        L_normalized = L * scale
        
        # Create distribution
        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L_normalized)
        dists.append(dist)
    return dists


def compute_hierarchical_logits(z, log_det, target_dists, semantic_counts, latent_pis):
    """
    Computes logits for the soft hierarchical bottleneck using Shared GMM.
    z_list: List of tensors [z0, z1, ...] from the model
    log_det: Tensor (B,)
    target_dists: List of MultivariateNormal batches (Shared Attributes)
    semantic_counts: List of integers [C0, C1, ...]
    latent_pis: List of (K, M_i) tensors (Class Mixture Weights)
    """
    logits = 0.0
    
    for i, z_part in enumerate(z):
        # Determine level. If we are at an intermediate step (aux loss), 
        # we treat the last part as the current level's latent.
        level = i
        if level >= len(semantic_counts):
            break
            
        n_semantic = semantic_counts[level]
        B, C, H, W = z_part.shape
        
        if n_semantic > 0:
            # Split into Noise and Texture
            # z_noise: (B, C - n_semantic, H, W)
            # z_texture: (B, n_semantic, H, W)
            z_noise = z_part[:, :-n_semantic]
            z_sem = z_part[:, -n_semantic:]
            
            # 1. Noise Log Prob (Standard Normal)
            if z_noise.numel() > 0:
                lp_noise = standard_normal_logprob(z_noise.reshape(B, -1)) # (B,)
            else:
                lp_noise = torch.zeros(B, device=z_part.device)
            
            # 2. Texture Log Prob (Shared GMM)
            # Flatten spatial dims: (B, n_sem, H, W) -> (B*H*W, n_sem)
            z_sem_flat = z_sem.permute(0, 2, 3, 1).reshape(-1, n_semantic)
            
            # Compute log_prob per pixel for each ATTRIBUTE cluster
            # dist has batch_shape (M,), event_shape (n_sem,)
            # input (N, 1, n_sem) broadcasts to (N, M)
            dist = target_dists[level]
            lp_sem_pixel_m = dist.log_prob(z_sem_flat.unsqueeze(1)) # (B*H*W, M)
            
            # Reshape back and sum over spatial dimensions
            # lp_sem_img_m: (B, M) - Log prob of image being in cluster m
            lp_sem_img_m = lp_sem_pixel_m.view(B, H, W, -1).sum(dim=(1, 2)) 
            
            # Combine with Class Mixture Weights
            # latent_pis[level]: (K, M)
            # We need to compute log p(z | y=k) = log sum_m pi_{k,m} p(z | m)
            # This results in a (B, K) matrix of logits
            
            # Expand for broadcasting:
            # lp_sem_img_m: (B, 1, M)
            # log_pis: (1, K, M)
            log_pis = torch.log_softmax(latent_pis[level], dim=-1)
            
            # weighted_log_prob: (B, K, M)
            weighted_log_prob = lp_sem_img_m.unsqueeze(1) + log_pis.unsqueeze(0)
            
            # LogSumExp over clusters (dim 2) -> (B, K)
            lp_sem_img = torch.logsumexp(weighted_log_prob, dim=2)
            
            # Accumulate
            logits = logits + lp_sem_img + lp_noise.unsqueeze(1)
        else:
            # Entire level is Noise
            z_noise = z_part
            lp_noise = standard_normal_logprob(z_noise.reshape(B, -1)) # (B,)
            logits = logits + lp_noise.unsqueeze(1)
        
    # Add Jacobian determinant
    logits = logits + log_det.unsqueeze(1)
    
    if logits.shape[1] == 1 and len(latent_pis) > 0:
        num_classes = latent_pis[0].shape[0]
        logits = logits.expand(-1, num_classes)
    
    return logits


def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss


def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss