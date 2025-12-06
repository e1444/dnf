import numpy as np
import torch
from torch import nn

def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)

def compute_logits(z, total_log_det, target_dists):
    log_phi_c = torch.stack([dist.log_prob(z) for dist in target_dists], dim=1)
    logits = log_phi_c + total_log_det.unsqueeze(1)
    return logits

def entropy_loss_fn(logits):
    probs = nn.functional.softmax(logits, dim=1)
    log_probs = nn.functional.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1).mean()
    return entropy

def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss

def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss

def total_loss_fn(logits, y_true, lambda_, label_smoothing=0.0):
    disc_loss = ce_loss_fn(logits, y_true, label_smoothing=label_smoothing)
    gen_loss = nll_loss_fn(logits, y_true)
    total_loss = (1 - lambda_) * disc_loss + lambda_ * gen_loss
    return total_loss

def deep_ce_loss(intermediate_logits, y_true, betas, label_smoothing=0.0):
    total_loss = torch.tensor(0.0, device=y_true.device)

    for j, logits_j in enumerate(intermediate_logits):
        ce_loss_j = ce_loss_fn(logits_j, y_true, label_smoothing=label_smoothing)
        total_loss += betas[j] * ce_loss_j

    return total_loss