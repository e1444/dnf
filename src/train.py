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
from src.utils.evaluation import evaluate, print_train_stats

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
    
    top_split = list(cfg.level_priors.priors.values())[-1].split    # top level split
    conf_features = top_split[0]  # noise features at top level
    for d in model.output_shapes[-1][1:]: # H, W of top level
        conf_features *= d
    
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors.values()):
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
                    rank=prior_cfg.rank
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
                    cond_features=conf_features,
                    H=H, W=W,
                    rank=prior_cfg.rank
                ).to(device)
                level_params.append(struct_prior)
            
            if sem_count > 0:
                if i < cfg.model.num_levels - 1:
                    cls = prior_cfg.conditional_cls
                    theta_list = hydra.utils.instantiate(
                        prior_cfg.random_init,
                        K=K,
                        C=sem_count, H=H, W=W,
                        rank=prior_cfg.rank
                    )
                    for theta in theta_list:
                        theta.update({
                            "z_channels": C,
                            "h_channels": sem_count,
                            "cond_features": conf_features,
                            "H": H,
                            "W": W
                        })
                else:
                    cls = prior_cfg.cls
                    theta_list = hydra.utils.instantiate(
                        prior_cfg.class_conditional_init,
                        K=K,
                        C=sem_count, H=H, W=W,
                        rank=prior_cfg.rank
                    )
                    
                sem_prior = ClassConditionalPrior([
                    hydra.utils.instantiate(cls, **theta) for theta in theta_list
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
    ema_model.output_shapes = model.output_shapes
        
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
            top_noise = splits[-1][0]
            z_style = outs[-1][1][:, :top_noise, :, :].reshape(x_batch.size(0), -1)
            for i, (z, h) in enumerate(outs):
                args = [
                    {},                         # noise params
                    {"z": z, "h": None},     # struct params
                    {}                          # semantic params
                ]
                if i < cfg.model.num_levels - 1:
                    args[2] = {"z": z, "h": z_style}  # semantic params
                split = splits[i]
                
                priors = [
                    prior_fact(**a) if prior_fact is not None else None 
                    for prior_fact, a in zip(level_priors[i], args)
                ]
                
                level_logits = compute_level_logits(z, h, priors, splits[i], K)
                logits = logits + level_logits
                    
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            task_loss = ce_loss
            
            # 2. NLL Loss (The Constraint)
            nll_loss = nll_loss_fn(logits, y_batch)
            
            # 3. Regularization terms
            reg_loss = cfg.training.r_logdet * (log_dets ** 2).mean()
            
            r_tau_density = float(cfg.training.r_tau_density)
            
            for i in range(cfg.model.num_levels):
                level_prior = level_priors[i]
                _, H, W = model.output_shapes[i]
                split = splits[i]
                
                # 1. Collect Shared Densities (Noise, Structure)
                shared_densities = []
                if level_prior[0] is not None:
                    dim = int(split[0])
                    shared_densities.append(level_prior[0].tau / (dim * H * W))
                if level_prior[1] is not None:
                    dim = int(split[1])
                    shared_densities.append(level_prior[1].tau / (dim * H * W))
                
                # 2. Collect Semantic Densities
                sem_densities = []
                if level_prior[2] is not None:
                    dim = int(split[2])
                    sem_densities = [p.tau / (dim * H * W) for p in level_prior[2].priors]
                
                # 3. Vectorized Variance Calculation
                if sem_densities:
                    # Stack semantic densities: (K,)
                    sem_tensor = torch.stack(sem_densities)
                    
                    if shared_densities:
                        # Stack shared densities: (M,)
                        shared_tensor = torch.stack(shared_densities)
                        
                        # Expand shared to (K, M) by repeating
                        shared_expanded = shared_tensor.unsqueeze(0).expand(len(sem_densities), -1)
                        
                        # Concatenate to (K, M+1): Each row is [Shared..., Sem_k]
                        path_tensor = torch.cat([shared_expanded, sem_tensor.unsqueeze(1)], dim=1)
                        
                        # Compute variance per row (per class path), then mean across classes
                        reg_tau_density = path_tensor.var(dim=1).mean()
                        reg_loss = reg_loss + r_tau_density * reg_tau_density
                    else:
                        # Only a single level => only a single scalar => undefined variance
                        pass
                        
                elif len(shared_densities) > 1:
                    # Only shared priors exist (no semantics)
                    reg_tau_density = torch.stack(shared_densities).var()
                    reg_loss = reg_loss + r_tau_density * reg_tau_density
            
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
            
            print_train_stats(epoch, train_stats, test_stats)

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
