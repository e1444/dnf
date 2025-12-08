import os
import hydra
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_dataset
from src.utils.losses import nll_loss_fn, ce_loss_fn, get_target_distributions, compute_hierarchical_logits
from src.utils.evaluation import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # Convert OmegaConf to a plain dictionary for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity, 
        config=config_dict, # type: ignore
        name=cfg.wandb.name
    )
    
    # Load data
    train_loader, test_loader = load_dataset(cfg.data)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    
    # Shared GMM Parameters
    num_classes = cfg.training.num_classes
    semantic_counts = cfg.training.semantic_counts
    attribute_counts = cfg.training.attribute_counts
    
    assert len(semantic_counts) == cfg.model.num_levels, \
        "Length of semantic_counts must match number of levels in the model."
    assert len(attribute_counts) == cfg.model.num_levels, \
        "Length of attribute_counts must match number of levels in the model."
    
    with torch.no_grad():
        attribute_means = nn.ParameterList()
        attribute_covs = nn.ParameterList()
        latent_pis = nn.ParameterList() # Class weights per level
        
        for num_attr, dim in zip(attribute_counts, semantic_counts):
            # Initialize Attribute Means (M, D)
            # Strategy: Assign roughly num_attr / K components to each class initially
            
            mu = torch.zeros(num_attr, dim, device=device)
            pis = torch.zeros(num_classes, num_attr, device=device)
            
            # 1. Generate Class Centers (K, D)
            class_centers = torch.randn(num_classes, dim, device=device)
            class_centers = class_centers / (class_centers.norm(dim=1, keepdim=True) + 1e-8)
            class_centers *= cfg.training.latent_separation
            
            # 2. Assign attributes to classes
            components_per_class = num_attr // num_classes
            remainder = num_attr % num_classes
            
            current_idx = 0
            for k in range(num_classes):
                n_comps = components_per_class + (1 if k < remainder else 0)
                
                # Initialize these components around the class center
                # Add noise to spread them out slightly around the center
                center = class_centers[k] # (D,)
                
                # (n_comps, D)
                # We use a smaller spread (0.2) so they stay relatively local to the class center
                cluster_means = center.unsqueeze(0) + torch.randn(n_comps, dim, device=device) * cfg.training.latent_mu_spread
                
                mu[current_idx : current_idx + n_comps] = cluster_means
                
                # Initialize weights: Strong preference for these components
                # We use logits for pi, so a large positive value means high probability
                pis[k, current_idx : current_idx + n_comps] = 3.0 # High logit
                # The rest remain 0.0 (neutral/low)
                
                current_idx += n_comps
            
            # Add global noise to break symmetries
            mu += torch.randn_like(mu) * 0.01
            
            # Set the parameters
            attribute_means.append(nn.Parameter(mu))
            
            # Covariances (Standard)
            L = torch.eye(dim, device=device).unsqueeze(0).repeat(num_attr, 1, 1)
            L *= cfg.training.latent_L_scale
            L += torch.randn_like(L) * 0.01
            attribute_covs.append(nn.Parameter(L))
            
            # Weights (with the bias we created)
            # Add noise to pis to allow gradient to flow even for "off" components
            pis += torch.randn_like(pis) * 0.1
            latent_pis.append(nn.Parameter(pis))
            
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_mu == 0:
        attribute_means.requires_grad_(False)
    if cfg.training.lr_L == 0:
        attribute_covs.requires_grad_(False)
    if cfg.training.lr_pi == 0:
        latent_pis.requires_grad_(False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr or 0,
        weight_decay=cfg.training.weight_decay or 0
    )
    optimizer.add_param_group({'params': attribute_means, 'lr': cfg.training.lr_mu, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': attribute_covs, 'lr': cfg.training.lr_L, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': latent_pis, 'lr': cfg.training.lr_pi, 'weight_decay': 0.0}) # Use lr_mu for weights
    
    # --- Augmented Lagrangian Setup ---
    log_alpha = torch.tensor(cfg.training.log_alpha, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=cfg.training.lr_log_alpha)
    rho = cfg.training.aug_rho
    nll_constraint = cfg.training.nll_constraint

    start_epoch = 0
    if cfg.training.resume_from_checkpoint is not None:
        print(f"Resuming training from {cfg.training.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.training.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load latent params
        if "attribute_means" in checkpoint:
            for i, p in enumerate(attribute_means): p.data.copy_(checkpoint['attribute_means'][i])
        elif "latent_mus" in checkpoint:
            for i, p in enumerate(attribute_means): p.data.copy_(checkpoint['latent_mus'][i])
        if "attribute_covs" in checkpoint:
            for i, p in enumerate(attribute_covs): p.data.copy_(checkpoint['attribute_covs'][i])
        
        if "latent_pis" in checkpoint:
            for i, p in enumerate(latent_pis): p.data.copy_(checkpoint['latent_pis'][i])
        
        if not cfg.training.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'log_alpha' in checkpoint: log_alpha.data.copy_(checkpoint['log_alpha'])
            if 'alpha_optimizer_state_dict' in checkpoint: alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    if cfg.training.reset_freeze:
        model.set_freeze_steps(freeze=False, start=0, end=-1)
    
    if cfg.training.freeze_steps.start is not None or cfg.training.freeze_steps.end is not None:
        model.set_freeze_steps(
            freeze=True,
            start=cfg.training.freeze_steps.start,
            end=cfg.training.freeze_steps.end
        )
        
    # Initialize EMA model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    steps_per_epoch = len(train_loader)
    total_warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    
    # Training loop
    print("Starting training...")
    total_epochs = start_epoch + cfg.training.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_ce = 0.0
        total_alpha = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if epoch < cfg.training.warmup_epochs:
                current_step = epoch * steps_per_epoch + batch_idx
                warmup_factor = (current_step + 1) / total_warmup_steps
                for param_group in optimizer.param_groups:
                    base_lr = cfg.training.lr_mu if any(param_group['params'][0] is t for t in attribute_means) else \
                              cfg.training.lr_L if any(param_group['params'][0] is t for t in attribute_covs) else \
                              cfg.training.lr
                    param_group['lr'] = base_lr * warmup_factor
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            z, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            # Check for NaNs in model output immediately
            if torch.isnan(log_det).any() or torch.isinf(log_det).any():
                print(f"CRITICAL ERROR: NaN/Inf detected in log_det at epoch {epoch} batch {batch_idx}.")
                return

            for i, z_part in enumerate(z):
                if torch.isnan(z_part).any() or torch.isinf(z_part).any():
                    print(f"CRITICAL ERROR: NaN/Inf detected in z[{i}] at epoch {epoch} batch {batch_idx}.")
                    return
            
            target_dists = get_target_distributions(attribute_means, attribute_covs)
            logits = compute_hierarchical_logits(z, log_det, target_dists, cfg.training.semantic_counts, latent_pis)
            
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            nll_loss = nll_loss_fn(logits, y_batch)
            reg_loss = cfg.training.r_logdet * (log_dets ** 2).mean()
            
            # Primal Objective
            primal_loss = ce_loss + reg_loss
            
            if torch.isnan(primal_loss) or torch.isinf(primal_loss):
                print(f"CRITICAL ERROR: NaN/Inf loss detected at epoch {epoch} batch {batch_idx}.")
                print("Terminating training to save resources.")
                return
            
            alpha = F.softplus(log_alpha)
            nll_violation = nll_loss - nll_constraint
            violation_pos = F.relu(nll_violation)
            
            if cfg.training.use_al:
                primal_loss += alpha * nll_violation 
                primal_loss += 0.5 * rho * violation_pos.pow(2)

            primal_loss.backward()
            
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradclip)
            torch.nn.utils.clip_grad_norm_(attribute_means, max_norm=cfg.training.gradclip_mu)
            torch.nn.utils.clip_grad_norm_(attribute_covs, max_norm=cfg.training.gradclip_L)
            torch.nn.utils.clip_grad_norm_(latent_pis, max_norm=cfg.training.gradclip_pi)
            
            optimizer.step()
            ema_model.update_parameters(model)

            # --- Dual Step ---
            if cfg.training.use_al:
                alpha_optimizer.zero_grad()
                dual_loss = -F.softplus(log_alpha) * nll_violation.detach()
                dual_loss.backward()
                alpha_optimizer.step()

            total_loss += primal_loss.item()
            total_nll += nll_loss.item()
            total_ce += ce_loss.item()
            total_alpha += alpha.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_alpha = total_alpha / len(train_loader)
        
        log_dict = {
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "train_nll": avg_nll,
            "train_ce": avg_ce,
            "alpha": avg_alpha,
            "nll_violation": avg_nll - nll_constraint
        }

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            eval_target_dists = get_target_distributions(attribute_means, attribute_covs)
            test_loss, test_accuracy, test_nll = evaluate(ema_model, test_loader, device, cfg, eval_target_dists, latent_pis)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_nll": test_nll
            })
            
            train_loss, train_accuracy, train_nll = evaluate(model, train_loader, device, cfg, eval_target_dists, latent_pis)
            log_dict.update({
                "train_eval_loss": train_loss,
                "train_eval_accuracy": train_accuracy,
                "train_eval_nll": train_nll
            })
            
            print(f"Epoch [{epoch+1:02d}/{total_epochs}] | Loss: {avg_train_loss:.4f} | Acc (Tr/Te): {train_accuracy:.2f}%/{test_accuracy:.2f}% | NLL (Tr/Te): {avg_nll:.2f}/{test_nll:.2f} | Alpha: {avg_alpha:.4f}")


        wandb.log(log_dict)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                'attribute_means': [p.detach().cpu() for p in attribute_means],
                'attribute_covs': [p.detach().cpu() for p in attribute_covs],
                'latent_pis': [p.detach().cpu() for p in latent_pis],
                'log_alpha': log_alpha.detach().cpu().clone(),
            }, checkpoint_path)
            # wandb.save(checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
