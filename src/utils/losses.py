import numpy as np
import torch
from torch import nn


def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)


def compute_level_logits(z, h, level_priors, level_split, K, sum=True):
    """
    Computes the logits for a single level of the hierarchy, using the formula:
    
    p(h^(i) | z^(i), y) = p(h^(i)_sem | y) p(h^(i)_struct | z^(i)) p(h^(i)_noise)
    
    and

    p(z^(L) | y) = p(z^(L)_sem | y) p(z^(L)_noise)
    
    Args:
        z: Tensor of shape (B, D, H, W) or None
        h: Tensor of shape (B, M, H, W)
        priors: tuple (noise_prior, struct_prior, sem_prior)
            noise_prior: callable or None
            struct_prior: callable or None
            sem_prior: callable or None
        level_split: tuple (noise_count, struct_count, sem_count)
    """
    noise_prior, struct_prior, sem_prior = level_priors
    noise_count, struct_count, sem_count = level_split
    
    # Split target variable h
    noise_h = h[:, :noise_count, :, :]
    struct_h = h[:, noise_count:noise_count+struct_count, :, :]
    sem_h = h[:, noise_count+struct_count:noise_count+struct_count+sem_count, :, :]
    
    B = h.shape[0]
    device = h.device
    dtype = h.dtype
    noise_lp = torch.zeros(B, K, device=device, dtype=dtype)
    struct_lp = torch.zeros(B, K, device=device, dtype=dtype)
    sem_lp = torch.zeros(B, K, device=device, dtype=dtype)

    # --- Noise Log-Prob ---
    if noise_prior is not None and noise_h is not None:
        noise_lp = noise_prior(unit_scale=True).log_prob(noise_h).unsqueeze(1).expand(-1, K)

    # --- Structural Log-Prob ---
    if struct_prior is not None and struct_h is not None:
        assert z is not None, "Structural prior requires z; struct_count > 0 on top level is invalid"
        struct_lp = struct_prior(z, unit_scale=True).log_prob(struct_h).unsqueeze(1).expand(-1, K)

    # --- Semantic Log-Probs (per class) ---
    if sem_prior is not None and sem_h is not None:
        sem_lp = torch.stack([dist.unit_scale and dist.log_prob(sem_h) or dist.log_prob(sem_h)  # keep unit_scale behavior from caller
                              for dist in sem_prior(unit_scale=True)], dim=1)

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