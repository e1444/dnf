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
    assert len(cfg.level_priors.priors) == cfg.model.num_levels, "Number of priors must match number of model levels"
    
    level_priors = []
    splits = []
    level_priors_params = nn.ModuleList()
    
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors):
            C, H, W = model.output_shapes[i]
            split = prior_cfg.split
            noise_count, struct_count, sem_count = split
            assert noise_count >= 0, "Noise feature dimension must be non-negative"
            assert struct_count >= 0, "Structure feature dimension must be non-negative"
            assert sem_count >= 0, "Semantic feature dimension must be non-negative"
            assert noise_count + struct_count + sem_count == C, "Sum of feature dimensions must equal total channels C"
            
            if i == cfg.model.num_levels - 1:
                assert struct_count == 0, "Top level cannot have structural features"
            
            noise_prior, struct_prior, sem_prior = None, None, None
            level_params = nn.ModuleList()
            
            if noise_count > 0:
                theta_list = hydra.utils.instantiate(
                    prior_cfg.zero_init, 
                    K=1,
                    C=noise_count, H=H, W=W,
                )
                noise_prior = hydra.utils.instantiate(
                    prior_cfg.cls,
                    **theta_list[0]
                ).to(device)
                level_params.append(noise_prior)
            
            if struct_count > 0:
                struct_prior = hydra.utils.instantiate(
                    prior_cfg.conditional_cls,
                    z_channels=C,
                    h_channels=struct_count,
                    H=H, W=W,
                ).to(device)
                level_params.append(struct_prior)
            
            if sem_count > 0:
                theta_list = hydra.utils.instantiate(
                    prior_cfg.class_conditional_init,
                    K=K,
                    C=sem_count, H=H, W=W,
                )
                sem_prior = ClassConditionalPrior([
                    hydra.utils.instantiate(prior_cfg.cls, **theta) for theta in theta_list
                ]).to(device)
                level_params.append(sem_prior)
                
            level_priors.append([noise_prior, struct_prior, sem_prior])
            level_priors_params.append(level_params)
            splits.append(split)
            
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    for prior_param, lr_prior in zip(level_priors_params, cfg.training.lr_prior):
        prior_param.requires_grad = (lr_prior > 0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    for prior_param, lr_prior in zip(level_priors_params, cfg.training.lr_prior):
        optimizer.add_param_group({
            'params': prior_param.parameters(),
            'lr': lr_prior,
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
                
                level_logits = compute_level_logits(z, h, priors, splits[k], K)
                logits = logits + level_logits
                
                r_noise, r_struct, r_sem = cfg.training.r_aniso
                if priors[0] is not None:
                    raniso = raniso + r_noise * priors[0].anisotropy_penalty()
                if priors[1] is not None:
                    raniso = raniso + r_struct * priors[1].anisotropy_penalty()
                if priors[2] is not None:
                    sem_penalty = sum(d.anisotropy_penalty() for d in priors[2]) / len(priors[2])
                    raniso = raniso + r_sem * sem_penalty
                    
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            task_loss = ce_loss
            
            # 2. NLL Loss (The Constraint)
            nll_loss = nll_loss_fn(logits, y_batch)
            
            # 3. Regularization terms
            reg_loss = cfg.training.r_logdet * (log_dets ** 2).mean()
            reg_loss = reg_loss + raniso
            
            # Primal Objective
            primal_loss = task_loss
            primal_loss += reg_loss
            
            if torch.isnan(primal_loss) or torch.isinf(primal_loss):
                print(f"WARNING: NaN/Inf loss detected at epoch {epoch} batch {batch_idx}.")
                print("Skipping batch.")
                optimizer.zero_grad()
                continue
            
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
            train_stats = evaluate(model, train_loader, device, cfg, level_priors, splits, prefix="train_eval")
            log_dict.update(train_stats)
            test_stats = evaluate(ema_model, test_loader, device, cfg, level_priors, splits, prefix="test")
            log_dict.update(test_stats)
            
            print(f"Epoch [{epoch+1:02d}/{total_epochs}] | Loss: {avg_train_loss:.4f} | Acc (Tr/Te): {train_stats['train_eval_accuracy']:.2f}%/{test_stats['test_accuracy']:.2f}% | NLL (Tr/Te): {train_stats['train_eval_nll']:.2f}/{test_stats['test_nll']:.2f} (Target {nll_constraint:.1f}) | Alpha: {avg_alpha:.4f}")
            
            split_names = ["noise", "structure", "semantics"]
            avg_logit_split = test_stats['test_logit_split']  # (3, L)
            level_labels = [f"level_{i}" for i in range(avg_logit_split.shape[1])]
            print("\nLogit split contributions (avg over samples):")
            print("          " + "  ".join(f"{lvl:>10}" for lvl in level_labels))
            for i, name in enumerate(split_names):
                row = "  ".join(f"{avg_logit_split[i, j]:>10.4f}" for j in range(avg_logit_split.shape[1]))
                print(f"{name:>10}  {row}")
            print("")

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
