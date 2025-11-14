import torch
from torch import nn

def compute_logits(z, total_log_det, target_dists):
    log_phi_c = torch.stack([dist.log_prob(z) for dist in target_dists], dim=1)
    logits = log_phi_c + total_log_det.unsqueeze(1)
    return logits

def entropy_loss_fn(logits):
    probs = nn.functional.softmax(logits, dim=1)
    log_probs = probs * nn.functional.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1).mean()
    return entropy

def disc_loss_fn(logits, y_true):
    disc_loss = nn.functional.cross_entropy(logits, y_true)
    return disc_loss

def generative_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss

def total_loss_fn(logits, y_true, lambda_):
    disc_loss = disc_loss_fn(logits, y_true)
    gen_loss = generative_loss_fn(logits, y_true)
    total_loss = disc_loss + lambda_ * gen_loss
    return total_loss

def deep_supervision_loss(intermediate_logits, y_true, alphas, betas):
    total_loss = torch.tensor(0.0, device=y_true.device)

    for j, logits_j in enumerate(intermediate_logits):
        disc_loss_j = disc_loss_fn(logits_j, y_true)
        ent_loss_j = entropy_loss_fn(logits_j)

        layer_loss = ent_loss_j + alphas[j] * disc_loss_j
        total_loss += betas[j] * layer_loss

    return total_loss