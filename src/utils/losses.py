import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_level_logits(z, h, spriors, nprior, sc):
    """
    Computes the logits for a single level of the hierarchy.
    
    Args:
        z: Conditioning variable (or None for final level).
        h: Target variable (split into semantic/noise).
        spriors: List of semantic priors (one per class).
        nprior: Non-semantic prior.
        sc: Number of semantic channels.
    """
    # Split target variable h into semantic (sh) and noise (nh)
    # Semantic is first sc channels, Noise is the rest
    sh = h[:, :sc, :, :]
    nh = h[:, sc:, :, :]
    
    # --- Non-Semantic Log-Prob ---
    if nprior is not None:
        if z is not None:
            mu, logs = nprior(z) # Conditional
        else:
            mu, logs = nprior()  # Learned (Unconditional)
            
        nprior_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logs))
        nlp = nprior_dist.log_prob(nh).sum(dim=[1, 2, 3])   # (B,)
    else:
        nlp = torch.zeros(h.shape[0], device=h.device)      # (B,)
    
    if sc == 0:
        K = len(spriors)
        return nlp.unsqueeze(1).expand(-1, K)  # (B, K)
    
    # --- Semantic Log-Probs (per class) ---
    slps = []
    for sprior in spriors:
        if sprior is None:
            # Should not happen if sc > 0, but safe to handle
            continue
        
        if z is not None:
            mu, logs = sprior(z) # Conditional
        else:
            mu, logs = sprior()  # Learned
            
        sprior_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logs))
        slp = sprior_dist.log_prob(sh).sum(dim=[1, 2, 3]) # (B,)
        slps.append(slp)
        
    slp = torch.stack(slps, dim=1) # (B, K)
    
    # Combine: Logits = Semantic_LogProb + NonSemantic_LogProb
    # Broadcasting nlp: (B,) -> (B, 1) to add to (B, K)
    return slp + nlp.unsqueeze(1)


def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss


def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss