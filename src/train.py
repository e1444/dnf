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


def split_prior_params_by_type(module: nn.Module):
    """Split prior parameters into channel, spatial, and other buckets."""
    channel_keys = ("cov_ch", "ch_", "ch_D_head", "ch_U_head", "log_cov_ch")
    spatial_keys = ("cov_sp", "sp_", "sp_D_head", "sp_U_head", "log_cov_sp")

    ch_params, sp_params, other_params = [], [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in channel_keys):
            ch_params.append(param)
        elif any(key in name for key in spatial_keys):
            sp_params.append(param)
        else:
            other_params.append(param)
    return ch_params, sp_params, other_params


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
                
            level_priors.append([noise_prior, struct_prior, sem_prior])
            level_priors_params.append(level_params)
            splits.append(split)
            
    # Initialize model
    model = hydra.utils.instantiate(
        cfg.model,
        input_shape=input_shape, 
        _convert_="partial"
    ).to(device)

    lr_prior_levels = list(cfg.training.lr_prior)
    lr_prior_channel = list(cfg.training.lr_prior_channel) if "lr_prior_channel" in cfg.training else lr_prior_levels
    lr_prior_spatial = list(cfg.training.lr_prior_spatial) if "lr_prior_spatial" in cfg.training else lr_prior_levels

    assert len(lr_prior_levels) == len(level_priors_params), "lr_prior length must match number of levels"
    assert len(lr_prior_channel) == len(level_priors_params), "lr_prior_channel length must match number of levels"
    assert len(lr_prior_spatial) == len(level_priors_params), "lr_prior_spatial length must match number of levels"

    if cfg.training.lr == 0:
        model.requires_grad_(False)
    for lr_base, lr_ch, lr_sp, level_prior_params in zip(lr_prior_levels, lr_prior_channel, lr_prior_spatial, level_priors_params):
        if lr_base == 0 and lr_ch == 0 and lr_sp == 0:
            level_prior_params.requires_grad_(False)

    optimizer_model = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    prior_param_groups = []
    for prior_param, lr_base, lr_ch, lr_sp in zip(level_priors_params, lr_prior_levels, lr_prior_channel, lr_prior_spatial):
        ch_params, sp_params, other_params = split_prior_params_by_type(prior_param)

        if lr_ch == 0:
            for p in ch_params:
                p.requires_grad_(False)
        if lr_sp == 0:
            for p in sp_params:
                p.requires_grad_(False)
        if lr_base == 0:
            for p in other_params:
                p.requires_grad_(False)

        if ch_params and lr_ch > 0:
            prior_param_groups.append({
                'params': ch_params,
                'lr': lr_ch,
                'weight_decay': 0
            })
        if sp_params and lr_sp > 0:
            prior_param_groups.append({
                'params': sp_params,
                'lr': lr_sp,
                'weight_decay': 0
            })
        if other_params and lr_base > 0:
            prior_param_groups.append({
                'params': other_params,
                'lr': lr_base,
                'weight_decay': 0
            })

    optimizer_prior = optim.AdamW(prior_param_groups) if len(prior_param_groups) > 0 else None
    trainable_prior_params = [p for p in level_priors_params.parameters() if p.requires_grad]
    
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
                if optimizer_prior is not None and 'optimizer_prior_state_dict' in checkpoint and checkpoint['optimizer_prior_state_dict'] is not None:
                    optimizer_prior.load_state_dict(checkpoint['optimizer_prior_state_dict'])
                elif optimizer_prior is None and 'optimizer_prior_state_dict' in checkpoint:
                    print("Warning: Prior optimizer not constructed; skipping optimizer_prior_state_dict.")
                elif optimizer_prior is not None:
                    print("Warning: optimizer_prior_state_dict missing in checkpoint.")
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
    
    # Compute dimension-aware NLL reweighting factors
    nll_dim_weights = [1.0] * cfg.model.num_levels  # Default: no reweighting
    if cfg.training.get('nll_dim_reweight', False):
        strategy = cfg.training.get('nll_dim_strategy', 'sqrt')
        print(f"Using dimension-aware NLL reweighting: {strategy}")
        
        dims = [C * H * W for C, H, W in output_shapes]
        dim_top = dims[-1]
        
        for i, dim in enumerate(dims):
            if strategy == "full":
                # Full normalization: 1/dim (all levels equal magnitude)
                nll_dim_weights[i] = dim_top / dim
            elif strategy == "sqrt":
                # Partial normalization: 1/sqrt(dim)
                nll_dim_weights[i] = (dim_top / dim) ** 0.5
            elif strategy == "inverse":
                # Inverse: dim_top / dim
                nll_dim_weights[i] = dim_top / dim
            # else: strategy == "none", keep weights at 1.0
        
        print(f"NLL dimension weights per level: {[f'{w:.4f}' for w in nll_dim_weights]}")
    
    # Training loop
    print("Starting training...")
    total_epochs = start_epoch + cfg.training.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        level_priors_params.train()
        
        total_loss = 0.0
        total_nll = 0.0
        total_ce = 0.0
        total_raw_nll = 0.0  # Unweighted NLL
        total_effective_nll = 0.0  # Weighted NLL used in loss
        level_raw_nlls = [0.0] * cfg.model.num_levels  # Per-level unweighted NLL
        level_effective_nlls = [0.0] * cfg.model.num_levels  # Per-level weighted NLL

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
            prior_logits_unweighted = torch.zeros_like(model_logits)  # Track unweighted version
            
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
                # Apply dimension-aware reweighting if enabled
                weighted_level_logits = level_logits * nll_dim_weights[k]
                prior_logits_acc = prior_logits_acc + weighted_level_logits
                prior_logits_unweighted = prior_logits_unweighted + level_logits
                
                # Track per-level NLL contributions
                with torch.no_grad():
                    C, H, W = output_shapes[k]
                    dim = C * H * W
                    level_nll_raw = -level_logits[torch.arange(x_batch.size(0)), y_batch].mean()
                    level_nll_raw_bpd = level_nll_raw / (dim * torch.log(torch.tensor(2.0)))
                    level_raw_nlls[k] += level_nll_raw_bpd.item()
                    
                    level_nll_eff = -weighted_level_logits[torch.arange(x_batch.size(0)), y_batch].mean()
                    level_nll_eff_bpd = level_nll_eff / (dim * torch.log(torch.tensor(2.0)))
                    level_effective_nlls[k] += level_nll_eff_bpd.item()
            
            # Detach priors for CE (Option A)
            # logits_ce = model_logits + prior_logits_acc.detach()
            # Option B: Allow priors to learn from CE (Discriminative Signal)
            logits_ce = model_logits + prior_logits_acc
            ce_loss = ce_loss_fn(logits_ce, y_batch, label_smoothing=cfg.training.label_smoothing)
            
            # Full logits for NLL
            logits = model_logits + prior_logits_acc
            logits_unweighted = model_logits + prior_logits_unweighted
            
            # 2. NLL Loss (this is the effective/weighted version used in optimization)
            nll_loss = nll_loss_fn(logits, y_batch)
            total_dim = input_shape[0] * input_shape[1] * input_shape[2]
            nll_loss = nll_loss / (total_dim * torch.log(torch.tensor(2.0)))
            
            # Track raw NLL (unweighted) for comparison
            with torch.no_grad():
                nll_loss_raw = nll_loss_fn(logits_unweighted, y_batch)
                nll_loss_raw_bpd = nll_loss_raw / (total_dim * torch.log(torch.tensor(2.0)))
                total_raw_nll += nll_loss_raw_bpd.item()
                total_effective_nll += nll_loss.item()
            
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
                    # Apply dimension-aware reweighting if enabled
                    weighted_level_logits = level_logits * nll_dim_weights[k]
                    prior_logits_acc = prior_logits_acc + weighted_level_logits
                    logits = model_logits + prior_logits_acc
                
                    nll_loss_em = nll_loss_fn(logits, y_batch)
                    nll_loss_em = nll_loss_em / (input_shape[0] * input_shape[1] * input_shape[2] * torch.log(torch.tensor(2.0)))
                    
                    nll_loss_em.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_prior_params, max_norm=cfg.training.gradclip)
                    optimizer_prior.step()
            
            # EMA update after EM
            ema_model.update_parameters(model)

        avg_train_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_raw_nll = total_raw_nll / len(train_loader)
        avg_effective_nll = total_effective_nll / len(train_loader)
        
        log_dict = {
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "train_nll": avg_nll,
            "train_ce": avg_ce,
            "train_nll_raw_bpd": avg_raw_nll,  # Unweighted NLL in bits per dimension
            "train_nll_effective_bpd": avg_effective_nll,  # Weighted NLL used in loss
        }
        
        # Add per-level NLL statistics
        for k in range(cfg.model.num_levels):
            log_dict[f"train_nll_raw_bpd_level_{k}"] = level_raw_nlls[k] / len(train_loader)
            log_dict[f"train_nll_effective_bpd_level_{k}"] = level_effective_nlls[k] / len(train_loader)

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
                'optimizer_prior_state_dict': optimizer_prior.state_dict() if optimizer_prior is not None else None,
            }, checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

if __name__ == "__main__":
    train()
