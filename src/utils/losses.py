import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_level_logits(z, h, level_priors, level_split):
    """
    Computes the logits for a single level of the hierarchy, using the formula:
    
    p(h^(i) | z^(i), y) = p(h^(i)_sem | z^(i), y) p(h^(i)_struct | z^(i)) p(h^(i)_noise)
    
    and

    p(z^(L)) = p(z^(L)_struct | z^(L)_sem) p(z^(L)_sem) p(z^(L)_noise)
    
    Args:
        z: Tensor of shape (B, D, H, W) or None
        h: Tensor of shape (B, M, H, W)
        noise_prior: Callable or None
        struct_prior: Callable or None
        sem_priors: List of Callables or None
        noise_count: int
        struct_count: int
        sem_count: int
    """
    noise_prior, struct_prior, sem_priors = level_priors
    noise_count, struct_count, sem_count = level_split
    
    # Split target variable h
    noise_h = h[:, :noise_count, :, :]
    struct_h = h[:, noise_count:noise_count+struct_count, :, :]
    sem_h = h[:, noise_count+struct_count:noise_count+struct_count+sem_count, :, :]
    
    total_lp = 0.0
    
    # --- Noise Log-Prob ---
    if noise_prior is not None:
        if z is not None:
            mu, logs = noise_prior()
        else:
            mu, logs = noise_prior()
            
        noise_prior_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logs))
        noise_lp = noise_prior_dist.log_prob(noise_h).sum(dim=[1, 2, 3])    # (B,)
    else:
        K = len(sem_priors)
        noise_lp = torch.zeros((h.shape[0], K), device=h.device)            # (B, K)
        
    total_lp = noise_lp
    
    # --- Structural Log-Prob ---
    if struct_prior is not None:
        if z is not None:
            mu, logs = struct_prior(z)
        else:
            mu, logs = struct_prior(sem_h)
            
        struct_prior_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logs))
        struct_lp = struct_prior_dist.log_prob(struct_h).sum(dim=[1, 2, 3]) # (B,)
        
        total_lp += struct_lp
        
    # --- Semantic Log-Probs (per class) ---
    sem_lps = []
    for sem_prior in sem_priors:
        if sem_prior is None:
            break
        
        if z is not None:
            mu, logs = sem_prior(z)
        else:
            mu, logs = sem_prior()
            
        sem_prior_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logs))
        sem_lp = sem_prior_dist.log_prob(sem_h).sum(dim=[1, 2, 3]) # (B,)
        sem_lps.append(sem_lp)
    
    if len(sem_lps) > 0:
        sem_lps = torch.stack(sem_lps, dim=1) # (B, K)
        total_lp += sem_lps
        
    return total_lp

def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss


def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss