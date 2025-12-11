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
from src.models.priors import ClassConditionalPrior
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
    input_shape = next(iter(train_loader))[0].shape[1:]  # (C, H, W)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model, input_shape=input_shape, _convert_="partial").to(device)
    
    K = cfg.data.dataset.num_classes
    noise_features = cfg.prior.noise_features
    struct_features = cfg.prior.struct_features
    semantic_features = cfg.prior.semantic_features
    splits = list(zip(noise_features, struct_features, semantic_features))
    
    assert len(noise_features) == cfg.model.num_levels, "Length of noise_features must match number of model levels"
    assert len(struct_features) == cfg.model.num_levels, "Length of struct_features must match number of model levels"
    assert len(semantic_features) == cfg.model.num_levels, "Length of semantic_features must match number of model levels"
    
    level_priors = []
    level_priors_params = nn.ModuleList()
    
    with torch.no_grad():
        for k in range(cfg.model.num_levels):
            C, H, W = model.output_shapes[k]
            noise_count, struct_count, sem_count = splits[k]
            assert noise_count >= 0, "Noise feature dimension must be non-negative"
            assert struct_count >= 0, "Structure feature dimension must be non-negative"
            assert sem_count >= 0, "Semantic feature dimension must be non-negative"
            assert noise_count + struct_count + sem_count == C, "Sum of feature dimensions must equal total channels C"
            
            if k == cfg.model.num_levels - 1:
                assert struct_count == 0, "Top level cannot have structural features"
            
            noise_prior, struct_prior, sem_prior = None, None, None
            
            if noise_count > 0:
                theta_list = hydra.utils.instantiate(cfg.prior.zero_init, K=1, C=noise_count, H=H, W=W)
                noise_prior = hydra.utils.instantiate(
                    cfg.prior.cls,
                    **theta_list[0]
                ).to(device)
                level_priors_params.append(noise_prior)
            
            if struct_count > 0:
                struct_prior = hydra.utils.instantiate(
                    cfg.prior.conditional_cls,
                    z_channels=C,
                    h_channels=struct_count,
                    H=H, W=W
                ).to(device)
                level_priors_params.append(struct_prior)
            
            if sem_count > 0:
                theta_list = hydra.utils.instantiate(cfg.prior.class_conditional_init, K=K, C=sem_count, H=H, W=W)
                sem_prior = ClassConditionalPrior([
                    hydra.utils.instantiate(cfg.prior.cls, **theta) for theta in theta_list
                ]).to(device)
                level_priors_params.append(sem_prior)
                
            level_priors.append((noise_prior, struct_prior, sem_prior))
            
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_prior == 0:
        level_priors_params.requires_grad_(False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    optimizer.add_param_group({
        'params': level_priors_params.parameters(),
        'lr': cfg.training.lr_prior,
        'weight_decay': cfg.training.weight_decay
    })
    
    # --- Augmented Lagrangian Setup ---
    log_alpha = torch.tensor(cfg.training.log_alpha, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=cfg.training.lr_log_alpha)
    
    rho = cfg.training.aug_rho
    nll_constraint = cfg.training.nll_constraint

    # Load from checkpoint if specified
    start_epoch = 0
    if cfg.training.resume_from_checkpoint is not None:
        print(f"Resuming training from {cfg.training.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.training.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
        
        if not cfg.training.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            log_alpha.data.copy_(checkpoint['log_alpha'])
            alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
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
                    if 'target_lr' not in param_group:
                        param_group['target_lr'] = param_group['lr']
                    
                    param_group['lr'] = param_group['target_lr'] * warmup_factor
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            
            logits = log_det.unsqueeze(1)
            for k, (z, h) in enumerate(outs):
                level_logits = compute_level_logits(z, h, level_priors[k], splits[k], K)
                logits = logits + level_logits
                
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
            torch.nn.utils.clip_grad_norm_(level_priors_params.parameters(), max_norm=cfg.training.gradclip)
            
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
            test_loss, test_accuracy, test_nll = evaluate(ema_model, test_loader, device, cfg, level_priors, splits)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_nll": test_nll
            })
            
            train_loss, train_accuracy, train_nll = evaluate(model, train_loader, device, cfg, level_priors, splits)
            log_dict.update({
                "train_eval_loss": train_loss,
                "train_eval_accuracy": train_accuracy,
                "train_eval_nll": train_nll
            })
            
            print(f"Epoch [{epoch+1:02d}/{total_epochs}] | Loss: {avg_train_loss:.4f} | Acc (Tr/Te): {train_accuracy:.2f}%/{test_accuracy:.2f}% | NLL (Tr/Te): {avg_nll:.2f}/{test_nll:.2f} (Target {nll_constraint:.1f}) | Alpha: {avg_alpha:.4f}")

        wandb.log(log_dict)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'prior_state_dict': level_priors_params.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                'log_alpha': log_alpha.detach().cpu().clone(),
            }, checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
