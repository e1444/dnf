import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.utils.losses import nll_loss_fn, ce_loss_fn, compute_level_logits


def evaluate(model, data_loader, device, cfg, level_priors, splits, output_shapes, prefix=None):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    for prior_level in level_priors:
        for prior in prior_level:
            if prior is not None:
                prior.eval()
                
    total_loss = 0.0
    total_nll = 0.0
    total_logit_split = 0.0
    total_log_det = 0.0
    correct = 0
    total = 0
    
    K = cfg.data.dataset.num_classes
    input_shape = None

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if input_shape is None:
                input_shape = x_batch.shape[1:]  # (C, H, W)
    
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            total_log_det = total_log_det + log_det.sum().item()
            
            # Compute logits using the shared utility
            logits = log_det.unsqueeze(1)   # (B, 1)
            all_level_logits = []
            anisotropy_losses = []
            for k, (h, z) in enumerate(outs):
                args = [{}, {"h": h}, {"h": h}]
                split = splits[k]
                
                priors = [
                    prior_fact(**a) if prior_fact is not None else None 
                    for prior_fact, a in zip(level_priors[k], args)
                ]
                
                for prior in priors:
                    if cfg.training.r_aniso <= 0.0:
                        pass
                    
                    if isinstance(prior, list):
                        for p in prior:
                            if p is not None:
                                anisotropy_losses.append(p.anisotropy_loss().mean())
                    elif prior is not None:
                        anisotropy_losses.append(K * prior.anisotropy_loss().mean())
                
                level_logits = compute_level_logits(h, z, priors, split, K, sum=False)     # (B, K, 3)
                all_level_logits.append(level_logits)
                level_logits_sum = torch.sum(level_logits, dim=2)  # (B, K)
                
                logits = logits + level_logits_sum
                
            all_level_logits = torch.stack(all_level_logits, dim=3)                             # (B, K, 3, L)
            total_logit_split = total_logit_split + torch.sum(all_level_logits, dim=(0, 1))     # Sum over B and K -> (3, L)
            
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            loss = ce_loss
            
            flow_log_dets = log_dets[1:]
            level_variances = flow_log_dets.var(dim=1)
            normalized_variances = []
            for i in range(len(level_variances)):
                # Get shape (C, H, W) for this level
                C, H, W = output_shapes[i]
                dim = C * H * W
                
                log_det_per_dim = flow_log_dets[i] / dim
                normalized_variances.append(log_det_per_dim.var())
                
            reg_loss = cfg.training.r_logdet * torch.stack(normalized_variances).mean()
            loss = loss + reg_loss
            
            if len(anisotropy_losses) > 0:
                reg_loss = reg_loss + cfg.training.r_aniso * torch.stack(anisotropy_losses).mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN/Inf loss detected during evaluation. Skipping batch.")
                continue
            
            total_loss += loss.item()
            
            # Calculate NLL for a clean evaluation metric
            nll_loss = nll_loss_fn(logits, y_batch)
            total_dim = input_shape[0] * input_shape[1] * input_shape[2]
            nll_loss = nll_loss / (total_dim * np.log(2.0))
            
            total_nll += nll_loss.item() * y_batch.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Handle case where all batches were skipped due to NaN
    if total == 0:
        print("WARNING: All evaluation batches were skipped due to NaN/Inf. Returning placeholder values.")
        if prefix is not None:
            prefix = prefix + '_'
        else:
            prefix = ''
        return {
            f"{prefix}loss": float('nan'),
            f"{prefix}accuracy": 0.0,
            f"{prefix}nll": float('nan'),
            f"{prefix}logit_split": float('nan'),
            f"{prefix}log_det": float('nan'),
        }
    
    avg_loss = total_loss / len(data_loader)
    avg_nll = total_nll / total
    avg_logit_split = total_logit_split / total
    avg_log_det = total_log_det / total
    accuracy = 100 * correct / total
    
    if prefix is not None:
        prefix = prefix + '_'
    else:
        prefix = ''
    
    return {
        f"{prefix}loss": avg_loss,
        f"{prefix}accuracy": accuracy,
        f"{prefix}nll": avg_nll,
        f"{prefix}logit_split": avg_logit_split,
        f"{prefix}log_det": avg_log_det,
    }
    

def print_split_table(title, split_tensor):
    # split_tensor: (3, L)
    split_names = ["noise", "structure", "semantics"]
    L = split_tensor.shape[1]
    level_labels = [f"level_{i}" for i in range(L)]

    row_sums = split_tensor.sum(dim=1)
    col_sums = split_tensor.sum(dim=0)
    total_sum = split_tensor.sum()

    print("\n" + "=" * 72)
    print(f"{title:^72}")
    print("=" * 72)

    # Header row
    print(f"{'Component':<15}" + "".join(f"{lvl:>12}" for lvl in level_labels) + f"{'Total':>12}")
    print("-" * 72)

    # Body rows
    for i, name in enumerate(split_names):
        vals = "".join(f"{split_tensor[i, j]:>12.1f}" for j in range(L))
        print(f"{name:<15}{vals}{row_sums[i]:>12.1f}")

    print("-" * 72)

    # Footer totals
    col_vals = "".join(f"{col_sums[j]:>12.1f}" for j in range(L))
    print(f"{'Total':<15}{col_vals}{total_sum:>12.1f}")

    print("=" * 72 + "\n")


def print_train_stats(epoch, train_stats, test_stats):
    print("\n" + "=" * 72)
    title = f"Evaluation Results (Epoch {epoch})"
    print(f"{title:^72}")
    print("=" * 72)
    
    print(f"{'Metric':<20}{'Train':>15}{'Test':>15}")
    print("-" * 72)
    
    print(f"{'Loss':<20}{train_stats['train_eval_loss']:>15.4f}{test_stats['test_loss']:>15.4f}")
    print(f"{'Accuracy (%)':<20}{train_stats['train_eval_accuracy']:>15.2f}{test_stats['test_accuracy']:>15.2f}")
    print(f"{'NLL':<20}{train_stats['train_eval_nll']:>15.2f}{test_stats['test_nll']:>15.2f}")
    print(f"{'Log-Det':<20}{train_stats['train_eval_log_det']:>15.2f}{test_stats['test_log_det']:>15.2f}")
    
    print("=" * 72 + "\n")

    print_split_table("Train Logit Split (Avg)", train_stats['train_eval_logit_split'])
    print_split_table("Test Logit Split (Avg)", test_stats['test_logit_split'])


def compute_marginal_bpd(model, loader, device, cfg, level_priors, splits):
    total_bpd = 0
    total_pixels = 0
    K = cfg.data.dataset.num_classes
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            # 1. Forward Pass
            outs, log_dets = model(x)
            log_det = torch.sum(log_dets, dim=0) # (B,)
            
            # 2. Compute Logits (log p(x|y))
            # We need log p(x|y) for each class y.
            # compute_level_logits returns (B, K, 3) where 3 is [noise, struct, sem]
            # Summing over the last dim gives log p(x|y) - log_det_jacobian
            
            logits = log_det.unsqueeze(1) # (B, 1)
            
            for k, (z, h) in enumerate(outs):
                args = [{}, {"z": z}, {}]
                split = splits[k]
                
                priors = [
                    prior_fact(**a) if prior_fact is not None else None 
                    for prior_fact, a in zip(level_priors[k], args)
                ]
                
                # (B, K, 3)
                level_logits = compute_level_logits(z, h, priors, split, K, sum=False)
                # Sum over components (noise+struct+sem) -> (B, K)
                logits = logits + torch.sum(level_logits, dim=2)
            
            # logits is now log p(x, y) = log p(x|y) + log p(y) if we assume uniform prior?
            # Actually compute_level_logits computes log p(z|y).
            # So logits = log p(z|y) + log |det J|.
            # This is log p(x|y).
            
            # 3. Marginalize: log p(x) = log sum_y p(x|y)p(y)
            # Assuming uniform prior p(y) = 1/K
            # log p(x) = log sum_y exp(log p(x|y)) - log K
            log_prob_marginal = torch.logsumexp(logits, dim=1) - np.log(K)
            
            # 4. Convert to Bits Per Dimension (BPD)
            # Formula: BPD = -log_2(p(x)) / D + 8
            n_pixels = x.shape[1] * x.shape[2] * x.shape[3]
            
            # Convert nats to bits (divide by ln 2)
            nll_bits = -log_prob_marginal / np.log(2)
            
            # Normalize by dimensions and add 8-bit offset
            bpd = (nll_bits / n_pixels) + 8.0
            
            total_bpd += bpd.sum().item()
            total_pixels += x.size(0)
            
    return total_bpd / total_pixels


def get_all_predictions(model, data_loader, device, cfg, level_priors, splits):
    """
    Get model predictions for an entire dataset.
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_probs = []
    all_confs = []
    
    K = cfg.data.dataset.num_classes

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            logits = log_det.unsqueeze(1)
            for k, (z, h) in enumerate(outs):
                args = [{}, {"z": z}, {}]
                split = splits[k]
                
                priors = [
                    prior_fact(**a) if prior_fact is not None else None 
                    for prior_fact, a in zip(level_priors[k], args)
                ]
                
                level_logits = compute_level_logits(z, h, priors, split, K, sum=True)
                logits = logits + level_logits

            probabilities = torch.softmax(logits, dim=1)
            confidences, y_pred = torch.max(probabilities, 1)

            all_y_true.append(y_batch.cpu())
            all_y_pred.append(y_pred.cpu())
            all_probs.append(probabilities.cpu())
            all_confs.append(confidences.cpu())

    return (
        torch.cat(all_y_true).numpy(),
        torch.cat(all_y_pred).numpy(),
        torch.cat(all_probs).numpy(),
        torch.cat(all_confs).numpy(),
    )


def get_classification_report_and_cm(y_true, y_pred, target_names):
    """
    Get classification report and plot confusion matrix.
    """
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return report, cm


def calculate_ece_and_reliability_diagram(confidences, predictions, true_labels, n_bins=15):
    """
    Calculate ECE and plot reliability diagram.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    accuracies_in_bin = []
    avg_conf_in_bin = []
    samples_in_bin_list = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        num_samples = np.sum(in_bin)
        samples_in_bin_list.append(num_samples)

        if prop_in_bin > 0:
            accuracy = np.mean(predictions[in_bin] == true_labels[in_bin])
            avg_confidence = np.mean(confidences[in_bin])
            ece += np.abs(accuracy - avg_confidence) * prop_in_bin
            accuracies_in_bin.append(accuracy)
            avg_conf_in_bin.append(avg_confidence)
        else:
            accuracies_in_bin.append(0)
            avg_conf_in_bin.append(0)
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # Print the data in a tabular format
    print("\n--- Reliability Diagram Data ---")
    bin_labels = [f"{low:.2f}-{high:.2f}" for low, high in zip(bin_lowers, bin_uppers)]
    data = {
        "Confidence Bin": bin_labels,
        "Avg Confidence": [f"{c:.4f}" for c in avg_conf_in_bin],
        "Accuracy": [f"{a:.4f}" for a in accuracies_in_bin],
        "Gap": [f"{abs(c-a):.4f}" for c, a in zip(avg_conf_in_bin, accuracies_in_bin)],
        "Samples": samples_in_bin_list,
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("--------------------------------\n")

    # Plot Reliability Diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    bin_centers = np.linspace(0, 1, n_bins * 2 + 1)[1::2]
    bin_width = 1.0 / n_bins
    plt.bar(bin_centers, accuracies_in_bin, width=bin_width, edgecolor='black', alpha=0.6, label='Accuracy')
    plt.bar(bin_centers, [c - a for c, a in zip(avg_conf_in_bin, accuracies_in_bin)], bottom=accuracies_in_bin, width=bin_width, edgecolor='black', alpha=0.4, color='red', label='Gap')
    plt.title('Reliability Diagram')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return ece


def calculate_brier_score(y_true, probabilities):
    """
    Calculate and Brier score.
    """
    y_true_one_hot = np.eye(probabilities.shape[1])[y_true]
    brier_score = np.mean(np.sum((probabilities - y_true_one_hot)**2, axis=1))
    print(f"Multi-class Brier Score: {brier_score:.4f}")
    return brier_score


def get_ood_confidences_and_plot(model, in_dist_loader, out_dist_loader, device, cfg, level_priors, splits):
    """
    Get and plot ROC curve and calculate AUROC for OOD detection.
    """
    model.eval()
    K = cfg.data.dataset.num_classes
    
    def get_confidences(loader):
        all_confs = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                
                outs, log_dets = model(x_batch)
                log_det = torch.sum(log_dets, dim=0)
                
                logits = log_det.unsqueeze(1)
                for k, (z, h) in enumerate(outs):
                    args = [{}, {"z": z}, {}]
                    split = splits[k]
                    
                    priors = [
                        prior_fact(**a) if prior_fact is not None else None 
                        for prior_fact, a in zip(level_priors[k], args)
                    ]
                    
                    level_logits = compute_level_logits(z, h, priors, split, K, sum=True)
                    logits = logits + level_logits
                
                probabilities = torch.softmax(logits, dim=1)
                confidences, _ = torch.max(probabilities, 1)
                all_confs.append(confidences.cpu())
        return torch.cat(all_confs).numpy()

    print("Computing In-Distribution confidences...")
    in_dist_confs = get_confidences(in_dist_loader)
    print("Computing Out-of-Distribution confidences...")
    out_dist_confs = get_confidences(out_dist_loader)

    # --- AUROC Calculation ---
    # Label 1: In-Distribution (Positive Class)
    # Label 0: Out-of-Distribution (Negative Class)
    y_true = np.concatenate([np.ones_like(in_dist_confs), np.zeros_like(out_dist_confs)])
    y_scores = np.concatenate([in_dist_confs, out_dist_confs])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    target_tpr = 0.95
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) > 0:
        fpr_at_95_tpr = fpr[idx[0]]
    else:
        fpr_at_95_tpr = 1.0 # Should not happen if curve goes to 1.0

    print(f"AUROC: {roc_auc:.4f}")
    print(f"FPR @ 95% TPR: {fpr_at_95_tpr:.4f}")
    print(f"Avg Confidence (ID): {np.mean(in_dist_confs):.4f}")
    print(f"Avg Confidence (OOD): {np.mean(out_dist_confs):.4f}")

    # --- Plotting ---
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (OOD classified as ID)')
    plt.ylabel('True Positive Rate (ID classified as ID)')
    plt.title('Receiver Operating Characteristic (OOD Detection)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc