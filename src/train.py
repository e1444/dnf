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
from src.utils.losses import nll_loss_fn, ce_loss_fn, deep_ce_loss, standard_normal_logprob
from src.utils.evaluation import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_target_distributions(latent_mu, latent_v, latent_U, cfg: DictConfig, device=torch.device('cpu')):
    K = cfg.training.num_classes
    D = cfg.training.features
    r = latent_U.shape[-1]
    
    latent_diag = torch.exp(latent_v) + cfg.training.latent_v_eps # (K, D)

    # Compute U^T D^{-1} U
    D_inv = (1.0 / latent_diag).unsqueeze(-1)         # (K, D, 1)
    U_scaled = latent_U * D_inv                      # (K, D, r)
    UT_Dinv_U = torch.bmm(latent_U.transpose(1, 2), U_scaled)  # (K, r, r)

    # M = I + U^T D^{-1} U
    I_r = torch.eye(r, device=device).unsqueeze(0).expand(K, -1, -1)
    M = I_r + UT_Dinv_U

    # Batched slogdet
    sign, logabsdet_M = torch.linalg.slogdet(M)

    # Fallback if any sign <= 0
    if torch.any(sign <= 0):
        M = M + 1e-6 * I_r
        _, logabsdet_M = torch.linalg.slogdet(M)

    # log det(diag) = sum(log diag) = sum(latent_v)
    logdet_D = torch.sum(latent_v, dim=1)

    cov_log_det = logdet_D + logabsdet_M  # shape (K,)

    # ----- Global scale normalization -----

    avg_log_det = cov_log_det.mean()      # scalar
    c = torch.exp(-avg_log_det / D)       # global scale
    sqrt_c = torch.sqrt(c)

    constrained_diag = latent_diag * c + cfg.training.latent_v_eps      # (K, D)
    constrained_U = latent_U * sqrt_c                                   # (K, D, r)

    return [
        torch.distributions.LowRankMultivariateNormal(
            loc=latent_mu[i],
            cov_factor=constrained_U[i],
            cov_diag=constrained_diag[i]
        ) for i in range(K)
    ]


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
        initial_mu = torch.zeros(cfg.training.num_classes, cfg.training.features, device=device)
        for i in range(cfg.training.num_classes):
            initial_mu[i, i] = cfg.training.latent_separation
        initial_mu += torch.randn_like(initial_mu) * cfg.training.latent_noise
        latent_mu = nn.Parameter(initial_mu)
        
        initial_v = torch.ones(cfg.training.num_classes, cfg.training.features, device=device) * torch.tensor(cfg.training.latent_v)
        initial_v += torch.randn_like(initial_v) * cfg.training.latent_noise
        latent_v = nn.Parameter(initial_v)
        
        initial_U = torch.zeros(cfg.training.num_classes, cfg.training.features, cfg.training.latent_U_size, device=device)
        latent_U = nn.Parameter(initial_U)
        
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_mu == 0:
        latent_mu.requires_grad_(False)
    if cfg.training.lr_v == 0:
        latent_v.requires_grad_(False)
    if cfg.training.lr_U == 0:
        latent_U.requires_grad_(False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr or 0,
        weight_decay=cfg.training.weight_decay or 0
    )
    optimizer.add_param_group({'params': [latent_mu], 'lr': cfg.training.lr_mu or 0, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [latent_v], 'lr': cfg.training.lr_v or 0, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [latent_U], 'lr': cfg.training.lr_U or 0, 'weight_decay': 0.0})
    
    # --- Augmented Lagrangian Setup ---
    # Initialize dual variable (log_alpha)
    log_alpha = torch.tensor(cfg.training.log_alpha, requires_grad=True, device=device)
    # Separate optimizer for alpha (dual)
    alpha_optimizer = optim.Adam([log_alpha], lr=cfg.training.lr_log_alpha)
    
    # Hyperparams
    rho = cfg.training.aug_rho
    # Default constraint: 1.0 nat per dimension if not specified
    nll_constraint = cfg.training.nll_constraint

    start_epoch = 0
    if cfg.training.resume_from_checkpoint is not None:
        print(f"Resuming training from {cfg.training.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.training.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        with torch.no_grad():
            if "latent_mu" in checkpoint:
                latent_mu.copy_(checkpoint['latent_mu'])
            if "latent_v" in checkpoint:
                latent_v.copy_(checkpoint['latent_v'])
            if "latent_U" in checkpoint:
                latent_U.copy_(checkpoint['latent_U'])
        
        if not cfg.training.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if cfg.training.lr is not None:
                optimizer.param_groups[0]['lr'] = cfg.training.lr
            else:
                cfg.training.lr = optimizer.param_groups[0]['lr']
                
            if cfg.training.weight_decay is not None:
                optimizer.param_groups[0]['weight_decay'] = cfg.training.weight_decay
                
            if cfg.training.lr_mu is not None:
                optimizer.param_groups[1]['lr'] = cfg.training.lr_mu
            else:
                cfg.training.lr_mu = optimizer.param_groups[1]['lr']
                
            if cfg.training.lr_v is not None:
                optimizer.param_groups[2]['lr'] = cfg.training.lr_v
            else:
                cfg.training.lr_v = optimizer.param_groups[2]['lr']
                
            if cfg.training.lr_U is not None:
                optimizer.param_groups[3]['lr'] = cfg.training.lr_U
            else:
                cfg.training.lr_U = optimizer.param_groups[3]['lr']
            
            # Load dual variables if available
            if 'log_alpha' in checkpoint:
                log_alpha.data.copy_(checkpoint['log_alpha'])
            if 'alpha_optimizer_state_dict' in checkpoint:
                alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                if cfg.training.lr_log_alpha is not None:
                    alpha_optimizer.param_groups[0]['lr'] = cfg.training.lr_log_alpha
        
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
        
    # Initialize scheduler
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
                    # Check param identity to assign correct specific LR
                    if param_group['params'][0] is latent_mu:
                        target_lr = cfg.training.lr_mu
                    elif param_group['params'][0] is latent_v:
                        target_lr = cfg.training.lr_v
                    elif param_group['params'][0] is latent_U:
                        target_lr = cfg.training.lr_U
                    else:
                        target_lr = cfg.training.lr
                    
                    param_group['lr'] = target_lr * warmup_factor
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            # Get the current dynamic target distributions
            target_dists = get_target_distributions(latent_mu, latent_v, latent_U, cfg, device)
                        
            # Compute logits
            log_prob_noise = 0.0
            for _, h in outs[:-1]:
                _log_prob_noise = standard_normal_logprob(h)
                log_prob_noise = log_prob_noise + _log_prob_noise
                
            z_semantic = outs[-1][1]
            log_prob_semantic = torch.stack([dist.log_prob(z_semantic) for dist in target_dists], dim=1)
            
            log_prob = log_prob_noise.unsqueeze(1) + log_prob_semantic
            logits = log_prob + log_det.unsqueeze(1)
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            task_loss = ce_loss
            
            # 2. NLL Loss (The Constraint)
            nll_loss = nll_loss_fn(logits, y_batch)
            
            # 3. Regularization terms
            reg_loss = cfg.training.r_logdet * (log_dets ** 2).mean()
            
            # Primal Objective
            primal_loss = task_loss
            primal_loss += reg_loss
            
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
            if latent_mu.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_mu], max_norm=cfg.training.gradclip_mu)
            if latent_v.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_v], max_norm=cfg.training.gradclip_v)
            if latent_U.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_U], max_norm=cfg.training.gradclip_U)
            
            optimizer.step()
            ema_model.update_parameters(model)

            # --- Dual Step ---
            if cfg.training.use_al:
                alpha_optimizer.zero_grad()
                # Maximize alpha * violation => Minimize -alpha * violation
                # Use detached violation to avoid affecting primal vars
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
            mean_v = latent_v.mean()
            constrained_v = latent_v - mean_v
            eval_target_dists = get_target_distributions(latent_mu, constrained_v, latent_U, cfg, device=device)
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
            
            print(f"Epoch [{epoch+1:02d}/{total_epochs}] | Loss: {avg_train_loss:.4f} | Acc (Tr/Te): {train_accuracy:.2f}%/{test_accuracy:.2f}% | NLL: {avg_nll:.2f} (Target {nll_constraint:.1f}) | Alpha: {avg_alpha:.4f}")


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
                'latent_mu': latent_mu.detach().cpu().clone(),
                'latent_v': latent_v.detach().cpu().clone(),
                'latent_U': latent_U.detach().cpu().clone(),
                'log_alpha': log_alpha.detach().cpu().clone(),
            }, checkpoint_path)
            # wandb.save(checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
