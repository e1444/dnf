import numpy as np
import torch
from torch import nn


def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)


def compute_level_logits(h, z, level_priors, level_split, K, sum=True):
    """
    Computes the logits for a single level of the hierarchy, using the formula:
    
    p(z^(i) | h^(i+1), y) = p(z^(i)_struct | h^(i+1)) p(z^(i)_noise)
    
    and

    p(h^(L) | y) = p(z^(L)_sem | y) p(z^(L)_noise)
    
    where
    
    h^(i) -> (h^(i+1), z^(i)_struct, z^(i)_noise)
    h^(L) -> (z^(L)_sem, z^(L)_noise)
    
    Args:
        z: Tensor of shape (B, D, H, W) or None
        h: Tensor of shape (B, M, H, W)
        level_priors: (noise_prior, struct_prior, sem_prior)
            noise_prior: Distribution or None
            struct_prior: Distribution or None
            sem_prior: List[Distribution] or None
        level_split: tuple (noise_count, struct_count, sem_count)
    """
    noise_prior, struct_prior, sem_prior = level_priors
    noise_count, struct_count, sem_count = level_split
    
    # Split target variable h
    noise_z = z[:, :noise_count, :, :]
    struct_z = z[:, noise_count:noise_count+struct_count, :, :]
    sem_z = z[:, noise_count+struct_count:noise_count+struct_count+sem_count, :, :]
    
    B = h.shape[0]
    device = h.device
    dtype = h.dtype
    noise_lp = torch.zeros(B, K, device=device, dtype=dtype)
    struct_lp = torch.zeros(B, K, device=device, dtype=dtype)
    sem_lp = torch.zeros(B, K, device=device, dtype=dtype)

    # --- Noise Log-Prob ---
    if noise_prior is not None and noise_z is not None:
        noise_lp = noise_prior.log_prob(noise_z).unsqueeze(1).expand(-1, K)

    # --- Structural Log-Prob ---
    if struct_prior is not None and struct_z is not None:
        assert h is not None, "Structural prior requires h; i.e. struct_count > 0 on top level is invalid"
        struct_lp = struct_prior.log_prob(struct_z).unsqueeze(1).expand(-1, K)

    # --- Semantic Log-Probs (per class) ---
    if sem_prior is not None and sem_z is not None:
        sem_lp = torch.stack([dist.log_prob(sem_z) for dist in sem_prior], dim=1)  # (B, K)
        
    if sum:
        return noise_lp + struct_lp + sem_lp  # (B, K)
    else:
        # (B, K, 3) with channels [noise, struct, sem]
        return torch.stack([noise_lp, struct_lp, sem_lp], dim=2)


def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss


def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss