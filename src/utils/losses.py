import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)


def get_target_distributions(latent_mus, latent_Ls):
    """
    Returns a list of MultivariateNormal distributions, one for each level.
    latent_mus: List of (K, D) tensors
    latent_Ls: List of (K, D, D) tensors (unconstrained)
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
        
        # 2. Enforce Unit Determinant (Volume Preservation)
        # log_det(L) = sum(log(diag))
        log_det_L = torch.sum(torch.log(diag), dim=-1, keepdim=True).unsqueeze(-1) # (K, 1, 1)
        scale = torch.exp(-log_det_L / D) # scalar factor
        
        L_normalized = L * scale
        
        # Create distribution
        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L_normalized)
        dists.append(dist)
    return dists


def compute_hierarchical_logits(z, log_det, target_dists, semantic_counts):
    """
    Computes logits for the soft hierarchical bottleneck.
    z_list: List of tensors [z0, z1, ...] from the model
    log_det: Tensor (B,)
    target_dists: List of MultivariateNormal batches
    semantic_counts: List of integers [C0, C1, ...]
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
            lp_noise = standard_normal_logprob(z_noise.reshape(B, -1)) # (B,)
            
            # 2. Texture Log Prob (Spatial Invariant Class Conditional)
            # Flatten spatial dims: (B, n_sem, H, W) -> (B*H*W, n_sem)
            z_sem_flat = z_sem.permute(0, 2, 3, 1).reshape(-1, n_semantic)
            
            # Compute log_prob per pixel for each class
            dist = target_dists[level]
            lp_sem_pixel = dist.log_prob(z_sem_flat.unsqueeze(1)) # (B*H*W, K)
            
            # Reshape back and sum over spatial dimensions
            lp_sem_img = lp_sem_pixel.view(B, H, W, -1).sum(dim=(1, 2)) # (B, K)
            
            # Accumulate
            logits = logits + lp_sem_img + lp_noise.unsqueeze(1)
        else:
            # Entire level is Noise
            z_noise = z_part
            lp_noise = standard_normal_logprob(z_noise.reshape(B, -1)) # (B,)
            logits = logits + lp_noise.unsqueeze(1)
        
    # Add Jacobian determinant
    logits = logits + log_det.unsqueeze(1)
    
    return logits


def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss


def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss