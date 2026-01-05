"""
Utilities for setting up training components (optimizers, checkpoints, etc.).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from omegaconf import DictConfig
from typing import Optional, Tuple, List


def split_prior_params_by_type(module: nn.Module) -> Tuple[List, List, List]:
    """
    Split prior parameters into channel, spatial, and other buckets.
    
    This allows differential learning rates for channel vs spatial covariance
    components in Kronecker-structured priors.
    
    Args:
        module: Prior module to split parameters from
        
    Returns:
        ch_params: Channel covariance parameters
        sp_params: Spatial covariance parameters  
        other_params: All other parameters (loc, tau, df, etc.)
    """
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


def setup_optimizers(
    model: nn.Module,
    level_priors_params: nn.ModuleList,
    cfg: DictConfig
) -> Tuple[optim.Optimizer, Optional[optim.Optimizer], List]:
    """
    Setup optimizers for model and priors with differential learning rates.
    
    Args:
        model: Flow model
        level_priors_params: ModuleList of prior parameters for each level
        cfg: Training configuration
        
    Returns:
        optimizer_model: Optimizer for flow parameters
        optimizer_prior: Optimizer for prior parameters (None if all LRs are 0)
        trainable_prior_params: List of trainable prior parameters
    """
    lr_prior_levels = list(cfg.training.lr_prior)
    lr_prior_channel = list(cfg.training.get('lr_prior_channel', lr_prior_levels))
    lr_prior_spatial = list(cfg.training.get('lr_prior_spatial', lr_prior_levels))

    assert len(lr_prior_levels) == len(level_priors_params), \
        f"lr_prior length ({len(lr_prior_levels)}) must match number of levels ({len(level_priors_params)})"
    assert len(lr_prior_channel) == len(level_priors_params), \
        f"lr_prior_channel length must match number of levels"
    assert len(lr_prior_spatial) == len(level_priors_params), \
        f"lr_prior_spatial length must match number of levels"

    # Freeze model if lr=0
    if cfg.training.lr == 0:
        model.requires_grad_(False)
        
    # Freeze entire prior levels if all their LRs are 0
    for lr_base, lr_ch, lr_sp, level_prior_params in zip(
        lr_prior_levels, lr_prior_channel, lr_prior_spatial, level_priors_params
    ):
        if lr_base == 0 and lr_ch == 0 and lr_sp == 0:
            level_prior_params.requires_grad_(False)

    # Setup model optimizer
    optimizer_model = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # Setup prior optimizer with parameter groups
    prior_param_groups = []
    for prior_param, lr_base, lr_ch, lr_sp in zip(
        level_priors_params, lr_prior_levels, lr_prior_channel, lr_prior_spatial
    ):
        ch_params, sp_params, other_params = split_prior_params_by_type(prior_param)

        # Freeze specific parameter types if their LR is 0
        if lr_ch == 0:
            for p in ch_params:
                p.requires_grad_(False)
        if lr_sp == 0:
            for p in sp_params:
                p.requires_grad_(False)
        if lr_base == 0:
            for p in other_params:
                p.requires_grad_(False)

        # Add parameter groups with their specific learning rates
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
    
    return optimizer_model, optimizer_prior, trainable_prior_params


def setup_ema_model(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    """
    Setup Exponential Moving Average model for evaluation.
    
    Args:
        model: Model to create EMA for
        decay: EMA decay rate
        
    Returns:
        ema_model: EMA-wrapped model
    """
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    ema_model.output_shapes = model.output_shapes
    return ema_model


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    level_priors_params: nn.ModuleList,
    ema_model: AveragedModel,
    optimizer_model: optim.Optimizer,
    optimizer_prior: Optional[optim.Optimizer],
    device: torch.device,
    load_model: bool = True,
    load_prior: bool = True,
    reset_optimizer: bool = False
) -> int:
    """
    Load training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Flow model to load state into
        level_priors_params: Prior parameters to load state into
        ema_model: EMA model to load state into
        optimizer_model: Model optimizer to load state into
        optimizer_prior: Prior optimizer to load state into (can be None)
        device: Device to map checkpoint to
        load_model: Whether to load model weights
        load_prior: Whether to load prior weights
        reset_optimizer: Whether to reset optimizer state
        
    Returns:
        start_epoch: Epoch to resume from
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if load_model:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Also update EMA to match loaded model
        if 'ema_model_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            # Fallback: load model state into the wrapped module
            ema_model.module.load_state_dict(checkpoint['model_state_dict'])
        ema_model.output_shapes = model.output_shapes
        
    if load_prior:
        level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
    
    if not reset_optimizer:
        if 'optimizer_model_state_dict' in checkpoint:
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            
            if optimizer_prior is not None and 'optimizer_prior_state_dict' in checkpoint:
                if checkpoint['optimizer_prior_state_dict'] is not None:
                    optimizer_prior.load_state_dict(checkpoint['optimizer_prior_state_dict'])
                else:
                    print("Warning: optimizer_prior_state_dict is None in checkpoint")
            elif optimizer_prior is None and 'optimizer_prior_state_dict' in checkpoint:
                print("Warning: Prior optimizer not constructed; skipping optimizer_prior_state_dict")
            elif optimizer_prior is not None:
                print("Warning: optimizer_prior_state_dict missing in checkpoint")
        else:
            print("Warning: Old checkpoint format detected. Optimizer state might not load correctly")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    return start_epoch


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    level_priors_params: nn.ModuleList,
    ema_model: AveragedModel,
    optimizer_model: optim.Optimizer,
    optimizer_prior: Optional[optim.Optimizer]
) -> str:
    """
    Save training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint in
        epoch: Current epoch number
        model: Flow model
        level_priors_params: Prior parameters
        ema_model: EMA model
        optimizer_model: Model optimizer
        optimizer_prior: Prior optimizer (can be None)
        
    Returns:
        checkpoint_path: Path where checkpoint was saved
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'prior_state_dict': level_priors_params.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict(),
        'optimizer_prior_state_dict': optimizer_prior.state_dict() if optimizer_prior is not None else None,
    }, checkpoint_path)
    return checkpoint_path
