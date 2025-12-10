import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from .losses import nll_loss_fn, ce_loss_fn, compute_level_logits


def evaluate(model, data_loader, device, cfg, priors, splits):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    total_test_loss = 0.0
    total_nll = 0.0
    correct = 0
    total_samples = 0
    
    semantic_counts = cfg.training.semantic_counts

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_size = x_batch.size(0)
            
            outs, log_dets = model(x_batch)
            
            # Compute Logits
            logits = 0
            for i in range(cfg.model.num_levels):
                z, h = outs[i]
                
                level_logits = compute_level_logits(z, h, priors[i], splits[i])
                logits = logits + level_logits
                    
            assert not isinstance(logits, int), "Logits computation failed; logits is None."
            
            log_det = torch.sum(log_dets, dim=0)
            
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            nll_loss = nll_loss_fn(logits + log_det.unsqueeze(1), y_batch)
            
            # reg_loss should be mean over batch to match ce_loss
            reg_loss = cfg.training.r_logdet * (log_dets ** 2).mean()
            
            primal_loss = ce_loss + reg_loss
            
            # Accumulate weighted by batch size
            total_test_loss += primal_loss.item() * batch_size
            total_nll += nll_loss.item() * batch_size
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_samples += batch_size
            correct += (predicted == y_batch).sum().item()
            
    avg_test_loss = total_test_loss / total_samples
    avg_nll = total_nll / total_samples
    accuracy = 100 * correct / total_samples
    
    return avg_test_loss, accuracy, avg_nll


def compute_marginal_bpd(model, loader, device, cfg, priors, splits):
    """
    Computes the Bits Per Dimension (BPD) by marginalizing over classes.
    log p(x) = log sum_c p(x|c)p(c)
    """
    total_nll_bits = 0.0
    total_dims = 0
    
    semantic_counts = cfg.training.semantic_counts
    num_classes = cfg.training.num_classes
    
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            # Forward Pass
            outs, log_dets = model(x)
            
            # Compute Logits
            logits = 0
            for i in range(cfg.model.num_levels):
                z, h = outs[i]
                
                level_logits = compute_level_logits(z, h, priors[i], splits[i])
                logits = logits + level_logits
                    
            assert not isinstance(logits, int), "Logits computation failed; logits is None."
            
            log_det = torch.sum(log_dets, dim=0)
            logits = logits + log_det.unsqueeze(1)
            
            # Marginalize: log p(x) = log sum_c p(x|c)p(c)
            # Assuming uniform prior p(c) = 1/K
            log_prob_marginal = torch.logsumexp(logits, dim=1) - np.log(num_classes)
            
            # NLL in bits
            # nll_bits = -log_prob_marginal / log(2)
            nll_bits = -log_prob_marginal / np.log(2)
            
            total_nll_bits += nll_bits.sum().item()
            total_dims += x.numel() # B * C * H * W
            
    # BPD = (Total NLL Bits / Total Dims) + 8 (for 8-bit data)
    bpd = (total_nll_bits / total_dims) + 8.0
    
    return bpd


def get_all_predictions(model, data_loader, device, cfg, priors, splits):
    """
    Get model predictions for an entire dataset.
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_probs = []
    all_confs = []
    
    semantic_counts = cfg.training.semantic_counts

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            
            outs, log_dets = model(x_batch)
            
            # Compute Logits
            logits = 0
            for i in range(cfg.model.num_levels):
                z, h = outs[i]
                
                level_logits = compute_level_logits(z, h, priors[i], splits[i])
                logits = logits + level_logits
                    
            assert not isinstance(logits, int), "Logits computation failed; logits is None."
            
            log_det = torch.sum(log_dets, dim=0)
            logits = logits + log_det.unsqueeze(1)
            
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

def get_ood_confidences_and_plot(model, in_dist_loader, out_dist_loader, device, cfg, priors, splits):
    """
    Get and plot ROC curve and calculate AUROC for OOD detection.
    """
    model.eval()
    
    semantic_counts = cfg.training.semantic_counts
    
    def get_confidences(loader):
        all_confs = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                
                outs, log_dets = model(x_batch)
                
                # Compute Logits
                logits = 0
                for i in range(cfg.model.num_levels):
                    z, h = outs[i]
                    
                    level_logits = compute_level_logits(z, h, priors[i], splits[i])
                    logits = logits + level_logits
                        
                assert not isinstance(logits, int), "Logits computation failed; logits is None."
                
                log_det = torch.sum(log_dets, dim=0)
                logits = logits + log_det.unsqueeze(1)
                
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

    print(f"AUROC: {roc_auc:.4f}")
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
