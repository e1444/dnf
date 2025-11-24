import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from .losses import deep_ce_loss, total_loss_fn, compute_logits, nll_loss_fn, ce_loss_fn, entropy_loss_fn

def evaluate(model, data_loader, device, cfg, target_dists, betas, lambda_, aux_layers):
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
            intermediate_outputs = [(z.view(z.size(0), -1), log_det) for z, log_det in intermediate_outputs]
            aux_outputs = [intermediate_outputs[i] for i in aux_layers]
            z, log_det = intermediate_outputs[-1]
            
            aux_logits = [
                compute_logits(z, log_det, target_dists) for z, log_det in aux_outputs
            ]
            
            logits = compute_logits(z, log_det, target_dists)
            
            # Calculate loss
            aux_loss = deep_ce_loss(aux_logits, y_batch, betas)
            final_loss = total_loss_fn(logits, y_batch, lambda_=lambda_)
            loss = aux_loss + final_loss
            
            # Regularization terms
            log_dets = [log_det for _, log_det in intermediate_outputs]
            for i in reversed(range(1, len(log_dets))):
                log_dets[i] = log_dets[i] - log_dets[i - 1]
                
            loss += cfg.training.r_logdet * (torch.stack(log_dets) ** 2).mean()
            total_test_loss += loss.item()
            
            # Calculate NLL for a clean evaluation metric
            nll_batch = nll_loss_fn(logits, y_batch) * y_batch.size(0)
            total_nll += nll_batch.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    avg_test_loss = total_test_loss / len(data_loader)
    avg_nll = total_nll / total
    accuracy = 100 * correct / total
    
    return avg_test_loss, accuracy, avg_nll

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
            
            intermediate_outputs = model(x_batch)
            z, log_det = intermediate_outputs[-1]
            z = z.view(z.size(0), -1)
            
            logits = compute_logits(z, log_det, target_dists)
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

def get_ood_confidences_and_plot(model, in_dist_loader, out_dist_loader, device, target_dists):
    """
    Get and plot confidences for in-distribution vs out-of-distribution data.
    """
    model.eval()
    
    def get_confidences(loader):
        all_confs = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                intermediate_outputs = model(x_batch)
                z, log_det = intermediate_outputs[-1]
                z = z.view(z.size(0), -1)
                logits = compute_logits(z, log_det, target_dists)
                probabilities = torch.softmax(logits, dim=1)
                confidences, _ = torch.max(probabilities, 1)
                all_confs.append(confidences.cpu())
        return torch.cat(all_confs).numpy()

    in_dist_confs = get_confidences(in_dist_loader)
    out_dist_confs = get_confidences(out_dist_loader)

    plt.figure(figsize=(12, 6))
    plt.hist(in_dist_confs, bins=50, alpha=0.7, label='In-Distribution', density=True)
    plt.hist(out_dist_confs, bins=50, alpha=0.7, label='Out-of-Distribution', density=True)
    plt.title('Model Confidence on In-Distribution vs. Out-of-Distribution Data')
    plt.xlabel('Confidence (Maximum Softmax Probability)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Average confidence on In-Distribution data: {np.mean(in_dist_confs):.4f}")
    print(f"Average confidence on Out-of-Distribution data: {np.mean(out_dist_confs):.4f}")
    return in_dist_confs, out_dist_confs
