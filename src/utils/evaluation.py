import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from .losses import nll_loss_fn, ce_loss_fn, compute_level_logits


def evaluate(model, data_loader, device, cfg, level_priors, splits, prefix=None):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_logit_split = 0.0
    correct = 0
    total = 0
    
    K = cfg.data.dataset.num_classes

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            # Compute logits using the shared utility
            logits = log_det.unsqueeze(1)   # (B, 1)
            all_level_logits = []
            raniso = 0.0
            for k, (z, h) in enumerate(outs):
                prior_facts = level_priors[k]
                priors = [None] * len(prior_facts)
                split = splits[k]
                
                if prior_facts[0] is not None:
                    priors[0] = prior_facts[0](unit_scale=True)        # noise_prior
                if prior_facts[1] is not None:
                    priors[1] = prior_facts[1](z, unit_scale=True)     # struct_prior
                if prior_facts[2] is not None:
                    priors[2] = prior_facts[2](unit_scale=True)        # sem_prior
                
                level_logits = compute_level_logits(z, h, priors, split, K, sum=False)     # (B, K, 3)
                all_level_logits.append(level_logits)
                logits = logits + torch.sum(level_logits, dim=2)
                
                r_noise, r_struct, r_sem = cfg.training.r_aniso
                if priors[0] is not None:
                    raniso = raniso + r_noise * priors[0].anisotropy_penalty()
                if priors[1] is not None:
                    raniso = raniso + r_struct * priors[1].anisotropy_penalty()
                if priors[2] is not None:
                    sem_penalty = sum(d.anisotropy_penalty() for d in priors[2]) / len(priors[2])
                    raniso = raniso + r_sem * sem_penalty
                
            all_level_logits = torch.stack(all_level_logits, dim=3)  # (B, K, 3, L)
            total_logit_split += torch.sum(all_level_logits, dim=(0, 1)).cpu().numpy()  # Sum over B and K -> (3, L)

            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            loss = ce_loss
            
            loss = loss + cfg.training.r_logdet * (log_dets ** 2).mean()
            loss = loss + raniso
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN/Inf loss detected during evaluation. Skipping batch.")
                continue
            
            total_loss += loss.item()
            
            # Calculate NLL for a clean evaluation metric
            nll_batch = nll_loss_fn(logits, y_batch) * y_batch.size(0)
            total_nll += nll_batch.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(data_loader)
    avg_nll = total_nll / total
    avg_logit_split = total_logit_split / total
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
    }


def compute_marginal_bpd(model, target_dists, loader, device, cfg):
    total_bpd = 0
    total_pixels = 0
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            # 2. Forward Pass on Dequantized Data
            zs = model(x)
            
            # 3. Preprocess (Flatten, Concat, Split)
            # Flatten each part of z: [B, C, H, W] -> [B, D_part]
            zs = [([z_part.view(z_part.size(0), -1) for z_part in z], log_det) for z, log_det in zs]
            # Concatenate: [Noise1 | Noise2 | ... | Semantic]
            zs = [(torch.cat(z, dim=1), log_det) for z, log_det in zs]
            # Slice: Noise = [:-feat], Semantic = [-feat:]
            feat = cfg.training.features
            zs = [((z[:, :-feat], z[:, -feat:]), log_det) for z, log_det in zs]
            
            # Get final state
            (z_noise, z_sem), log_det = zs[-1]
            
            # 4. Calculate Noise Probability: log p(z_noise) ~ N(0, I)
            # -0.5 * (z^2 + log(2pi)) summed over dimensions
            log_p_noise = -0.5 * (z_noise ** 2 + np.log(2 * np.pi)).sum(dim=1)
            
            # 5. Calculate log p(x|c) for ALL classes
            log_probs_conditional = []
            num_classes = len(target_dists)
            
            for c in range(num_classes):
                target = target_dists[c]
                # log p(z_sem | c)
                log_p_sem_given_c = target.log_prob(z_sem)
                
                # Total log p(x | c) = log p(noise) + log p(sem | c) + log_det
                # Note: log_p_noise and log_det are shared across classes
                log_p_x_given_c = log_p_noise + log_p_sem_given_c + log_det
                log_probs_conditional.append(log_p_x_given_c)
            
            log_probs_conditional = torch.stack(log_probs_conditional, dim=1)
            
            # 6. Marginalize: log p(x) = log sum_c p(x|c)p(c)
            # Assuming uniform prior p(c) = 1/K
            # log p(x) = log sum exp(log p(x|c)) - log(K)
            log_prob_marginal = torch.logsumexp(log_probs_conditional, dim=1) - np.log(num_classes)
            
            # 7. Convert to Bits Per Dimension (BPD)
            # Formula: BPD = -log_2(p(x)) / D + 8
            n_pixels = x.shape[1] * x.shape[2] * x.shape[3]
            
            # Convert nats to bits (divide by ln 2)
            nll_bits = -log_prob_marginal / np.log(2)
            
            # Normalize by dimensions and add 8-bit offset
            bpd = (nll_bits / n_pixels) + 8.0
            
            total_bpd += bpd.sum().item()
            total_pixels += x.size(0)
            
    return total_bpd / total_pixels


def get_all_predictions(model, data_loader, device, target_dists):
    """
    Get model predictions for an entire dataset.
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_probs = []
    all_confs = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            
            zs = model(x_batch)
            zs = [([z_part.view(z_part.size(0), -1) for z_part in z], log_det) for z, log_det in zs] # Flatten each part of z
            z, log_det = zs[-1]
            z = (torch.cat(z[:-1], dim=1), z[-1])  # Concatenate all but semantic part
            
            noise_dists = torch.distributions.Independent(
                torch.distributions.Normal(loc=torch.zeros_like(z[0]), scale=torch.ones_like(z[0])), 
                reinterpreted_batch_ndims=1
            )
            
            log_prob_noise = noise_dists.log_prob(z[0])
            log_prob_semantic = torch.stack([dist.log_prob(z[1]) for dist in target_dists], dim=1)
            log_prob = log_prob_noise.unsqueeze(1) + log_prob_semantic
            logits = log_prob + log_det.unsqueeze(1)
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

def get_ood_confidences_and_plot(model, in_dist_loader, out_dist_loader, device, cfg, target_dists):
    """
    Get and plot ROC curve and calculate AUROC for OOD detection.
    """
    model.eval()
    
    def get_confidences(loader):
        all_confs = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                
                # --- Standardized Preprocessing (Matches get_all_predictions) ---
                zs = model(x_batch)
                # Flatten parts
                zs = [([z_part.view(z_part.size(0), -1) for z_part in z], log_det) for z, log_det in zs]
                # Concatenate all parts
                zs = [(torch.cat(z, dim=1), log_det) for z, log_det in zs]
                # Slice: Noise = [:-feat], Semantic = [-feat:]
                feat = cfg.training.features
                zs = [((z[:, :-feat], z[:, -feat:]), log_det) for z, log_det in zs]
                
                # Get final state
                (z_noise, z_sem), log_det = zs[-1]
                
                # --- Compute Logits ---
                noise_dists = torch.distributions.Independent(
                    torch.distributions.Normal(loc=torch.zeros_like(z_noise[0]), scale=torch.ones_like(z_noise[0])), 
                    reinterpreted_batch_ndims=1
                )
                
                log_prob_noise = noise_dists.log_prob(z_noise)
                log_prob_semantic = torch.stack([dist.log_prob(z_sem) for dist in target_dists], dim=1)
                
                # Broadcast noise prob
                log_prob = log_prob_noise.unsqueeze(1) + log_prob_semantic
                logits = log_prob + log_det.unsqueeze(1)
                
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
