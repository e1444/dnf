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
from src.utils.prior_builder import build_level_priors_from_config
from src.utils.training_setup import (
    setup_optimizers,
    setup_ema_model,
    load_checkpoint,
    save_checkpoint
)
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
    
    # Initialize priors
    input_shape = next(iter(train_loader))[0].shape[1:]  # (C, H, W)
    
    if cfg.model._target_ == "src.models.glow.DGLOWNetwork":
        from src.models.glow import DGLOWNetwork
        output_shapes = DGLOWNetwork.output_shapes(input_shape, cfg.model.num_levels)
    else:
        raise NotImplementedError(f"Model {cfg.model._target_} not supported.")
    
    K = cfg.data.dataset.num_classes
    
    # Initialize priors from configuration
    level_priors, splits, level_priors_params = build_level_priors_from_config(
        cfg=cfg,
        output_shapes=output_shapes,
        num_classes=K,
        device=device
    )
    
    # Initialize model
    model = hydra.utils.instantiate(
        cfg.model,
        input_shape=input_shape, 
        _convert_="partial"
    ).to(device)

    # Setup optimizers
    optimizer_model, optimizer_prior, trainable_prior_params = setup_optimizers(
        model=model,
        level_priors_params=level_priors_params,
        cfg=cfg
    )
    
    # Setup EMA model
    ema_model = setup_ema_model(model, decay=cfg.training.get('ema_decay', 0.999))
    
    # Load from checkpoint if specified
    start_epoch = 0
    if cfg.training.ckpt is not None:
        start_epoch = load_checkpoint(
            checkpoint_path=cfg.training.ckpt,
            model=model,
            level_priors_params=level_priors_params,
            ema_model=ema_model,
            optimizer_model=optimizer_model,
            optimizer_prior=optimizer_prior,
            device=device,
            load_model=cfg.training.load_model,
            load_prior=cfg.training.load_prior,
            reset_optimizer=cfg.training.reset_optimizer
        )
    else:
        # For new training: Initialize EMA with current model state
        # This ensures ActNorm and other stateful layers are properly initialized
        ema_model.module.load_state_dict(model.state_dict())
        ema_model.output_shapes = model.output_shapes
        
    # Initialize scheduler
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer_model)
    steps_per_epoch = len(train_loader)
    total_warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    
    r_logdet = cfg.training.r_logdet
    
    # Training loop
    print("Starting training...")
    total_epochs = start_epoch + cfg.training.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        level_priors_params.train()
        
        total_loss = 0.0
        total_nll = 0.0
        total_ce = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            if epoch < cfg.training.warmup_epochs:
                current_step = epoch * steps_per_epoch + batch_idx
                warmup_factor = (current_step + 1) / total_warmup_steps
                
                for optimizer in [opt for opt in [optimizer_model, optimizer_prior] if opt is not None]:
                    for param_group in optimizer.param_groups:
                        if 'target_lr' not in param_group:
                            param_group['target_lr'] = param_group['lr']
                        param_group['lr'] = param_group['target_lr'] * warmup_factor
                    
                # r_logdet = cfg.training.r_logdet * (1 - warmup_factor)
            
            optimizer_model.zero_grad()
            if optimizer_prior is not None:
                optimizer_prior.zero_grad()

            # Forward pass
            outs, log_dets = model(x_batch)
            log_det = torch.sum(log_dets, dim=0)
            model_logits = log_det.unsqueeze(1)
            prior_logits_acc = torch.zeros_like(model_logits)
            
            anisotropy_losses = []
            for k, (h, z) in enumerate(outs):
                args = [{}, {"h": h}, {}]
                split = splits[k]
                
                priors = [
                    prior_fact(**a) if prior_fact is not None else None 
                    for prior_fact, a in zip(level_priors[k], args)
                ]
                
                for prior in priors:
                    if cfg.training.r_aniso <= 0.0:
                        pass
                    
                    if isinstance(prior, list):
                        for p in prior:
                            if p is not None:
                                anisotropy_losses.append(p.anisotropy_loss().mean())
                    elif prior is not None:
                        anisotropy_losses.append(K * prior.anisotropy_loss().mean())
                        
                level_logits = compute_level_logits(h, z, priors, splits[k], K)
                prior_logits_acc = prior_logits_acc + level_logits
            
            # Detach priors for CE (Option A)
            # logits_ce = model_logits + prior_logits_acc.detach()
            # Option B: Allow priors to learn from CE (Discriminative Signal)
            logits_ce = model_logits + prior_logits_acc
            ce_loss = ce_loss_fn(logits_ce, y_batch, label_smoothing=cfg.training.label_smoothing)
            
            # Full logits for NLL
            logits = model_logits + prior_logits_acc
            
            # 2. NLL Loss
            nll_loss = nll_loss_fn(logits, y_batch)
            total_dim = input_shape[0] * input_shape[1] * input_shape[2]
            nll_loss = nll_loss / (total_dim * torch.log(torch.tensor(2.0)))
            
            task_loss = (1 - cfg.training.l_lambda) * ce_loss + cfg.training.l_lambda * nll_loss
            
            # 3. Regularization terms
            # 3.1. Log-Det Variance Regularization
            flow_log_dets = log_dets[1:]
            for i in range(cfg.model.num_levels):
                # Get shape (C, H, W) for this level
                C, H, W = output_shapes[i]
                dim = C * H * W
                
                flow_log_dets[i] = flow_log_dets[i] / dim
            
            reg_loss = r_logdet * (flow_log_dets ** 2).mean()
            
            # 3.2. Anisotropy Regularization
            if len(anisotropy_losses) > 0:
                reg_loss = reg_loss + cfg.training.r_aniso * torch.stack(anisotropy_losses).mean()
            
            # Primal Objective
            task_loss = task_loss + reg_loss
            
            # Check for NaN/Inf in loss before backprop
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                print(f"WARNING: NaN/Inf in task_loss at batch {batch_idx}")
                print(f"  CE loss: {ce_loss.item():.2e}, NLL loss: {nll_loss.item():.2e}, Reg loss: {reg_loss.item():.2e}")
                print(f"  Logits stats: min={logits.min().item():.2e}, max={logits.max().item():.2e}")
                print(f"  Model logits stats: min={model_logits.min().item():.2e}, max={model_logits.max().item():.2e}")
                print(f"  Prior logits stats: min={prior_logits_acc.min().item():.2e}, max={prior_logits_acc.max().item():.2e}")
                continue
            
            task_loss.backward()
            
            # Check for NaN/Inf in gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"WARNING: NaN/Inf gradient in model parameter: {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Skipping batch {batch_idx} due to NaN gradients")
                optimizer_model.zero_grad()
                if optimizer_prior is not None:
                    optimizer_prior.zero_grad()
                continue
            
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradclip)
            if len(trainable_prior_params) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_prior_params, max_norm=cfg.training.gradclip)
            
            optimizer_model.step()
            if optimizer_prior is not None:
                optimizer_prior.step()

            total_loss += task_loss.item()
            total_nll += nll_loss.item()
            total_ce += ce_loss.item()
            
            # EM step
            if optimizer_prior is not None and len(trainable_prior_params) > 0:
                for _ in range(cfg.training.em_steps):
                    optimizer_prior.zero_grad()
                    
                    with torch.no_grad():
                        outs, log_dets = model(x_batch)
                        log_det = torch.sum(log_dets, dim=0)
                        model_logits = log_det.unsqueeze(1)
                    
                    prior_logits_acc = torch.zeros_like(model_logits)
                    for k, (h, z) in enumerate(outs):
                        args = [{}, {"h": h}, {}]
                        split = splits[k]
                        
                        priors = [
                            prior_fact(**a) if prior_fact is not None else None 
                            for prior_fact, a in zip(level_priors[k], args)
                        ]
                        
                        level_logits = compute_level_logits(h, z, priors, splits[k], K)
                        prior_logits_acc = prior_logits_acc + level_logits
                    
                    logits = model_logits + prior_logits_acc
                
                    nll_loss_em = nll_loss_fn(logits, y_batch)
                    nll_loss_em = nll_loss_em / (input_shape[0] * input_shape[1] * input_shape[2] * torch.log(torch.tensor(2.0)))
                    
                    nll_loss_em.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_prior_params, max_norm=cfg.training.gradclip)
                    optimizer_prior.step()
            
            # EMA update after EM (only if use_ema is enabled)
            if cfg.training.use_ema:
                ema_model.update_parameters(model)

        avg_train_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        
        # Debug: Check for NaN in model and prior parameters
        if cfg.training.ckpt is None and epoch == start_epoch:
            print("DEBUG: Checking parameters for NaN/Inf after first epoch...")
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  Model NaN/Inf in: {name}")
                    has_nan = True
            for name, param in level_priors_params.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  Prior NaN/Inf in: {name}")
                    has_nan = True
            if not has_nan:
                print("  No NaN/Inf found in parameters")
        
        log_dict = {
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "train_nll": avg_nll,
            "train_ce": avg_ce,
        }

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            train_stats = evaluate(model, train_loader, device, cfg, level_priors, splits, output_shapes, prefix="train_eval")
            log_dict.update(train_stats)
            
            # Use EMA model if enabled, otherwise use regular model
            eval_model = ema_model if cfg.training.use_ema else model
            test_stats = evaluate(eval_model, test_loader, device, cfg, level_priors, splits, output_shapes, prefix="test")
            log_dict.update(test_stats)
            
            print_train_stats(epoch, train_stats, test_stats)

        wandb.log(log_dict)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
            checkpoint_path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                level_priors_params=level_priors_params,
                ema_model=ema_model,
                optimizer_model=optimizer_model,
                optimizer_prior=optimizer_prior
            )
            print(f"Saved checkpoint to {checkpoint_path}")
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
