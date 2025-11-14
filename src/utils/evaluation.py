import torch
import torch.nn as nn
import numpy as np
from .losses import deep_supervision_loss, total_loss_fn, compute_logits

def evaluate(model, data_loader, device, cfg, target_dists, alphas, betas, lambda_, aux_layers):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    total_test_loss = 0.0
    total_nll = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            intermediate_outputs = model(x_batch)
            aux_outputs = [intermediate_outputs[i] for i in aux_layers]
            z, log_det = intermediate_outputs[-1]
            
            aux_logits = [
                compute_logits(z, log_det, target_dists) for z, log_det in aux_outputs
            ]
            
            logits = compute_logits(z, log_det, target_dists)
            
            # Calculate loss
            aux_loss = deep_supervision_loss(aux_logits, y_batch, alphas, betas)
            final_loss = total_loss_fn(logits, y_batch, lambda_=lambda_)
            loss = aux_loss + final_loss
            
            # Regularization terms
            loss += cfg.training.r_logdet * (log_det ** 2).mean()
            
            # Note: The trainable_log_vars are part of the optimizer, not directly in the model
            # Regularization for trainable_log_vars should be handled in the training loop
            
            total_test_loss += loss.item()
            
            # Calculate NLL for a clean evaluation metric
            nll_batch = nn.functional.cross_entropy(logits, y_batch, reduction='sum')
            total_nll += nll_batch.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    avg_test_loss = total_test_loss / len(data_loader)
    avg_nll = total_nll / total
    accuracy = 100 * correct / total
    
    return avg_test_loss, accuracy, avg_nll
