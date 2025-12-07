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
    
    with torch.no_grad():
        latent_mus = nn.ParameterList()
        latent_Ls = nn.ParameterList()
        
        for dim in cfg.training.semantic_counts:
            # Initialize Mu
            mu = torch.zeros(cfg.training.num_classes, dim, device=device)
            mu += torch.randn_like(mu) * cfg.training.latent_separation
            mu += torch.randn_like(mu) * cfg.training.latent_noise
            latent_mus.append(nn.Parameter(mu))
            
            # Initialize L (Identity + Noise)
            L = torch.eye(dim, device=device).unsqueeze(0).repeat(cfg.training.num_classes, 1, 1)
            L *= cfg.training.latent_L_scale
            L += torch.randn_like(L) * 0.01
            latent_Ls.append(nn.Parameter(L))
        
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_mu == 0:
        latent_mus.requires_grad_(False)
    if cfg.training.lr_L == 0:
        latent_Ls.requires_grad_(False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr or 0,
        weight_decay=cfg.training.weight_decay or 0
    )
    optimizer.add_param_group({'params': latent_mus, 'lr': cfg.training.lr_mu or 0, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': latent_Ls, 'lr': cfg.training.lr_L or 0, 'weight_decay': 0.0})
    
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
        if "latent_mus" in checkpoint:
            for i, p in enumerate(latent_mus): p.data.copy_(checkpoint['latent_mus'][i])
        if "latent_Ls" in checkpoint:
            for i, p in enumerate(latent_Ls): p.data.copy_(checkpoint['latent_Ls'][i])
        
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
                    base_lr = cfg.training.lr_mu if param_group['params'][0] in latent_mus else \
                              cfg.training.lr_L if param_group['params'][0] in latent_Ls else \
                              cfg.training.lr
                    param_group['lr'] = base_lr * warmup_factor
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            z, log_det = model(x_batch)
            target_dists = get_target_distributions(latent_mus, latent_Ls)
            logits = compute_hierarchical_logits(z, log_det, target_dists, cfg.training.semantic_counts)
            
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            nll_loss = nll_loss_fn(logits, y_batch)
            reg_loss = cfg.training.r_logdet * (log_det ** 2).mean()
            
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
            torch.nn.utils.clip_grad_norm_(latent_mus, max_norm=cfg.training.gradclip_mu)
            torch.nn.utils.clip_grad_norm_(latent_Ls, max_norm=cfg.training.gradclip_L)
            
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
            eval_target_dists = get_target_distributions(latent_mus, latent_Ls)
            test_loss, test_accuracy, test_nll = evaluate(ema_model, test_loader, device, cfg, eval_target_dists)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_nll": test_nll
            })
            
            train_loss, train_accuracy, train_nll = evaluate(model, train_loader, device, cfg, eval_target_dists)
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
                'latent_mus': [p.detach().cpu() for p in latent_mus],
                'latent_Ls': [p.detach().cpu() for p in latent_Ls],
                'log_alpha': log_alpha.detach().cpu().clone(),
            }, checkpoint_path)
            # wandb.save(checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
