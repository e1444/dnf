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
from src.models.prior import ConditionalPrior, LearnedPrior, mu_simplex_init
from src.utils.losses import nll_loss_fn, ce_loss_fn, compute_level_logits
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
        name=cfg.wandb.name,
        group=cfg.wandb.group
    )
    
    # Load data
    train_loader, test_loader = load_dataset(cfg.data)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    num_classes = cfg.training.num_classes
    
    assert len(cfg.training.noise_counts) == cfg.model.num_levels, \
        "Length of noise_counts must match number of levels in the model."
    assert len(cfg.training.struct_counts) == cfg.model.num_levels, \
        "Length of struct_counts must match number of levels in the model."
    assert len(cfg.training.semantic_counts) == cfg.model.num_levels, \
        "Length of semantic_counts must match number of levels in the model."
    
    splits = [(noise, struct, sem) for noise, struct, sem in zip(
        cfg.training.noise_counts, 
        cfg.training.struct_counts, 
        cfg.training.semantic_counts
    )]
    priors = []
    prior_params = nn.ModuleList()
    for i in range(cfg.model.num_levels):
        shape = model.output_shapes[i]
        noise_count, struct_count, sem_count = splits[i]
        C, H, W = shape
        
        assert noise_count + struct_count + sem_count == C, \
            f"At level {i}, sum of noise_counts, struct_counts, and semantic_counts must equal {C}."
        assert noise_count >= 0 and struct_count >= 0 and sem_count >= 0, \
            f"At level {i}, counts must be non-negative."
        
        noise_prior, struct_prior, sem_priors = None, None, [None] * num_classes
        if i < cfg.model.num_levels - 1:
            if noise_count > 0:
                noise_prior = LearnedPrior(shape=(noise_count, H, W), cov_method="diag")
                prior_params.append(noise_prior)
            if struct_count > 0:
                struct_prior = ConditionalPrior(in_channels=C, out_channels=struct_count, cov_method="diag")
                prior_params.append(struct_prior)
                
            if sem_count > 0:
                init_mus = mu_simplex_init(num_classes, sem_count, scale=cfg.training.simplex_scale)
                sem_priors = []
                for k in range(num_classes):
                    sem_prior = ConditionalPrior(in_channels=C, out_channels=sem_count, init_mu=init_mus[k], cov_method="diag")
                    prior_params.append(sem_prior)
                    sem_priors.append(sem_prior)
        else:
            if noise_count > 0:
                noise_prior = LearnedPrior(shape=(noise_count, H, W), cov_method="diag")
                prior_params.append(noise_prior)
            if struct_count > 0:
                assert sem_count > 0, "Final level with structural channels must also have semantic channels."
                struct_prior = ConditionalPrior(in_channels=sem_count, out_channels=struct_count, cov_method="diag")
                prior_params.append(struct_prior)
            if sem_count > 0:
                init_mus = mu_simplex_init(num_classes, sem_count, scale=cfg.training.simplex_scale)
                sem_priors = []
                for k in range(num_classes):
                    sem_prior = LearnedPrior(shape=(sem_count, H, W), init_mu=init_mus[k], cov_method="diag")
                    prior_params.append(sem_prior)
                    sem_priors.append(sem_prior)
                
        priors.append((noise_prior, struct_prior, sem_priors))
    
    prior_params.to(device)
    
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_prior == 0:
        prior_params.requires_grad_(False)
        
    if cfg.training.optimizer == "Adamax":
        optimizer_cls = optim.Adamax
    else:
        optimizer_cls = optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.training.lr or 0,
        weight_decay=cfg.training.weight_decay or 0
    )
    optimizer.add_param_group({
        'params': prior_params.parameters(),
        'lr': cfg.training.lr_prior,
        'weight_decay': cfg.training.weight_decay
    })
    
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
        prior_params.load_state_dict(checkpoint['priors_state_dict'])
        
        if not cfg.training.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'log_alpha' in checkpoint: log_alpha.data.copy_(checkpoint['log_alpha'])
            if 'alpha_optimizer_state_dict' in checkpoint: alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
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
                    if 'target_lr' not in param_group:
                        param_group['target_lr'] = param_group['lr']
                    
                    param_group['lr'] = param_group['target_lr'] * warmup_factor
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            # Check for NaNs in model output immediately
            if torch.isnan(log_det).any() or torch.isinf(log_det).any():
                print(f"CRITICAL ERROR: NaN/Inf detected in log_det at epoch {epoch} batch {batch_idx}.")
                return

            for i, out in enumerate(outs):
                z, h = out
                if z is not None and (torch.isnan(z).any() or torch.isinf(z).any()):
                    print(f"CRITICAL ERROR: NaN/Inf detected in z[{i}] at epoch {epoch} batch {batch_idx}.")
                    return
                
                if torch.isnan(h).any() or torch.isinf(h).any():
                    print(f"CRITICAL ERROR: NaN/Inf detected in h[{i}] at epoch {epoch} batch {batch_idx}.")
                    return
            
            logits = 0
            for i in range(cfg.model.num_levels):
                z, h = outs[i]
                
                level_logits = compute_level_logits(z, h, priors[i], splits[i])
                logits = logits + level_logits
                    
            assert not isinstance(logits, int), "Logits computation failed; logits is None."
            
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            nll_loss = nll_loss_fn(logits + log_det.unsqueeze(1), y_batch)
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
            torch.nn.utils.clip_grad_norm_(prior_params.parameters(), max_norm=cfg.training.gradclip_prior)
            
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
            test_loss, test_accuracy, test_nll = evaluate(ema_model, test_loader, device, cfg, priors, splits)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_nll": test_nll
            })
            
            train_loss, train_accuracy, train_nll = evaluate(model, train_loader, device, cfg, priors, splits)
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
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "priors_state_dict": prior_params.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
                "log_alpha": log_alpha.detach().cpu().clone(),
            }, checkpoint_path)
            # wandb.save(checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
