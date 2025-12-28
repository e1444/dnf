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
    
    # Initialize priors
    input_shape = next(iter(train_loader))[0].shape[1:]  # (C, H, W)
    
    if cfg.model._target_ == "src.models.glow.DGLOWNetwork":
        from src.models.glow import DGLOWNetwork
        output_shapes = DGLOWNetwork.output_shapes(input_shape, cfg.model.num_levels)
    else:
        raise NotImplementedError(f"Model {cfg.model._target_} not supported.")
    
    std_per_level = []
    
    K = cfg.data.dataset.num_classes
    assert len(cfg.level_priors.priors) == cfg.model.num_levels, "Number of priors must match number of model levels"
    
    level_priors = []
    splits = []
    level_priors_params = nn.ModuleList()
    
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors.values()):
            C, H, W = output_shapes[i]
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
            
            std = torch.zeros(C, device=device)
            std_idx = 0
            
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
                
                # tau = D * ln(sigma^2) => sigma = exp(tau / 2D)
                D_noise = noise_count * H * W
                noise_std = torch.exp(torch.tensor(prior_cfg.cls.tau) / (2 * D_noise))
                std[std_idx:std_idx + noise_count] = noise_std
                std_idx += noise_count
            
            if struct_count > 0:
                if prior_cfg.conditional_cls._target_ == "src.models.priors.ConditionalMixturePrior":
                    num_components = prior_cfg.conditional_cls.num_components
                    del prior_cfg.conditional_cls.num_components
                    theta_list = hydra.utils.instantiate(
                        prior_cfg.class_conditional_init,
                        K=num_components,
                        C=struct_count, H=H, W=W,
                        rank=prior_cfg.rank
                    )
                    components = [
                        hydra.utils.instantiate(prior_cfg.cls, **theta) for theta in theta_list
                    ]
                    struct_prior = hydra.utils.instantiate(
                        prior_cfg.conditional_cls,
                        components=components,
                        h_channels=C
                    ).to(device)
                    level_params.append(struct_prior)
                elif prior_cfg.conditional_cls._target_ in [
                    "src.models.priors.ConditionalKPMVNPrior",
                    "src.models.priors.ConditionalKPMVTPrior"
                ]:
                    struct_prior = hydra.utils.instantiate(
                        prior_cfg.conditional_cls,
                        h_channels=struct_count,
                        z_channels=C,
                        H=H, W=W,
                        rank=prior_cfg.rank
                    ).to(device)
                    level_params.append(struct_prior)
                else:
                    raise NotImplementedError(f"Conditional prior {prior_cfg.conditional_cls._target_} not supported.")
                
                D_struct = struct_count * H * W
                struct_std = torch.exp(torch.tensor(prior_cfg.conditional_cls.tau) / (2 * D_struct))
                std[std_idx:std_idx + struct_count] = struct_std
                std_idx += struct_count
            
            if sem_count > 0:
                theta_list = hydra.utils.instantiate(
                    prior_cfg.class_conditional_init,
                    K=K,
                    C=sem_count, H=H, W=W,
                    rank=prior_cfg.rank
                )
                sem_prior = ClassConditionalPrior([
                    hydra.utils.instantiate(prior_cfg.cls, **theta) for theta in theta_list
                ]).to(device)
                level_params.append(sem_prior)
                
                D_sem = sem_count * H * W
                sem_std = torch.exp(torch.tensor(prior_cfg.class_conditional_init.tau_marginal) / (2 * D_sem))
                std[std_idx:std_idx + sem_count] = sem_std
                std_idx += sem_count
                
            level_priors.append([noise_prior, struct_prior, sem_prior])
            level_priors_params.append(level_params)
            splits.append(split)
            
            std_per_level.append(std)
            
    # Initialize model
    model = hydra.utils.instantiate(
        cfg.model,
        input_shape=input_shape, 
        std_per_level=std_per_level,
        _convert_="partial"
    ).to(device)
    
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    for lr_prior, level_prior_params in zip(cfg.training.lr_prior, level_priors_params):
        if lr_prior == 0:
            level_prior_params.requires_grad_(False)

    optimizer_model = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    
    prior_param_groups = []
    for prior_param, lr_prior in zip(level_priors_params, cfg.training.lr_prior):
        params_to_optimize = [p for p in prior_param.parameters() if p.requires_grad]
        if len(params_to_optimize) > 0:
            prior_param_groups.append({
                'params': params_to_optimize,
                'lr': lr_prior,
                'weight_decay': 0
            })
    optimizer_prior = optim.AdamW(prior_param_groups)
    
    # Load from checkpoint if specified
    start_epoch = 0
    if cfg.training.ckpt is not None:
        print(f"Resuming training from {cfg.training.ckpt}")
        checkpoint = torch.load(cfg.training.ckpt, map_location=device)
        
        if cfg.training.load_model:
            model.load_state_dict(checkpoint['model_state_dict'])
        if cfg.training.load_prior:
            level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
        
        if not cfg.training.reset_optimizer:
            if 'optimizer_model_state_dict' in checkpoint:
                optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
                optimizer_prior.load_state_dict(checkpoint['optimizer_prior_state_dict'])
            else:
                print("Warning: Old checkpoint format detected. Optimizer state might not load correctly.")
        
        start_epoch = checkpoint['epoch'] + 1
        
    # Initialize EMA model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
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
                
                for optimizer in [optimizer_model, optimizer_prior]:
                    for param_group in optimizer.param_groups:
                        if 'target_lr' not in param_group:
                            param_group['target_lr'] = param_group['lr']
                        
                        param_group['lr'] = param_group['target_lr'] * warmup_factor
                    
                # r_logdet = cfg.training.r_logdet * (1 - warmup_factor)
            
            optimizer_model.zero_grad()
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
            logits_ce = model_logits + prior_logits_acc.detach()
            ce_loss = ce_loss_fn(logits_ce, y_batch, label_smoothing=cfg.training.label_smoothing)
            
            # Full logits for NLL
            logits = model_logits + prior_logits_acc
            
            # 2. NLL Loss
            nll_loss = nll_loss_fn(logits, y_batch)
            nll_loss = nll_loss / (input_shape[0] * input_shape[1] * input_shape[2] * torch.log(torch.tensor(2.0)))
            
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
            
            task_loss.backward()
            
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradclip)
            torch.nn.utils.clip_grad_norm_(level_priors_params.parameters(), max_norm=cfg.training.gradclip)
            
            optimizer_model.step()
            optimizer_prior.step()

            total_loss += task_loss.item()
            total_nll += nll_loss.item()
            total_ce += ce_loss.item()
            
            # EM step
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
                torch.nn.utils.clip_grad_norm_(level_priors_params.parameters(), max_norm=cfg.training.gradclip)
                optimizer_prior.step()
            
            # EMA update after EM
            ema_model.update_parameters(model)

        avg_train_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        
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
            test_stats = evaluate(ema_model, test_loader, device, cfg, level_priors, splits, output_shapes, prefix="test")
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
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'optimizer_prior_state_dict': optimizer_prior.state_dict(),
            }, checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
