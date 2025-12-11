import numpy as np
import torch
from torch import nn

def standard_normal_logprob(z_tensor):
    return -0.5 * (z_tensor.pow(2) + np.log(2 * np.pi)).sum(dim=1)

def ce_loss_fn(logits, y_true, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, y_true, label_smoothing=label_smoothing)
    return ce_loss

def nll_loss_fn(logits, y_true):
    batch_size = logits.shape[0]
    true_class_logits = logits[torch.arange(batch_size), y_true]
    gen_loss = -true_class_logits.mean()
    return gen_loss