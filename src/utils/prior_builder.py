"""
High-level prior construction from Hydra configs.

This module provides utilities for building the hierarchical prior structure
for VS-Flow from configuration files.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import List, Tuple

from src.utils.prior_init import (
    create_unconditional_priors,
    create_conditional_prior,
    create_class_conditional_prior,
    create_conditional_mixture_prior,
)


def build_level_priors_from_config(
    cfg: DictConfig,
    output_shapes: List[Tuple[int, int, int]],
    num_classes: int,
    device: torch.device
) -> Tuple[List[List], List[Tuple[int, int, int]], nn.ModuleList]:
    """
    Build hierarchical priors for all levels from configuration.
    
    This implements the VS-Flow prior structure:
        p(z) = p(z_sem | y) * p(z_struct^(L)) * ∏ p(z_struct^(i) | h^(i+1))
    
    Args:
        cfg: Hydra configuration containing model and level_priors
        output_shapes: List of (C, H, W) shapes for each level's latent z
        num_classes: Number of classes K
        device: Device to create priors on
    
    Returns:
        level_priors: List of [noise_prior, struct_prior, sem_prior] for each level
        splits: List of (noise_count, struct_count, sem_count) tuples
        level_priors_params: ModuleList containing all trainable prior parameters
    """
    num_levels = cfg.model.num_levels
    assert len(cfg.level_priors.priors) == num_levels, \
        f"Number of priors ({len(cfg.level_priors.priors)}) must match num_levels ({num_levels})"
    
    level_priors = []
    splits = []
    level_priors_params = nn.ModuleList()
    
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors.values()):
            C, H, W = output_shapes[i]
            split = prior_cfg.split
            noise_count, struct_count, sem_count = split
            
            # Validate split configuration
            assert noise_count >= 0, f"Level {i}: Noise feature dimension must be non-negative"
            assert struct_count >= 0, f"Level {i}: Structure feature dimension must be non-negative"
            assert sem_count >= 0, f"Level {i}: Semantic feature dimension must be non-negative"
            assert noise_count + struct_count + sem_count == C, \
                f"Level {i}: Sum of split {split} must equal total channels {C}"
            
            # Top level cannot have structural features (no h^(i+1) to condition on)
            if i == num_levels - 1:
                assert struct_count == 0, \
                    f"Top level (L={num_levels-1}) cannot have structural features"
            
            noise_prior, struct_prior, sem_prior = None, None, None
            level_params = nn.ModuleList()
            
            # ============================================================
            # Noise Prior: p(z_noise) - unconditional, captures pure randomness
            # ============================================================
            if noise_count > 0:
                noise_prior = _build_noise_prior(
                    prior_cfg=prior_cfg,
                    noise_count=noise_count,
                    H=H, W=W,
                    device=device
                )
                level_params.append(noise_prior)
            
            # ============================================================
            # Structural Prior: p(z_struct^(i) | h^(i+1))
            # Conditional on next level to ensure hierarchical consistency
            # ============================================================
            if struct_count > 0:
                struct_prior = _build_struct_prior(
                    prior_cfg=prior_cfg,
                    struct_count=struct_count,
                    C=C,  # h^(i+1) has C channels (same as z due to 50/50 split)
                    H=H, W=W,
                    device=device
                )
                level_params.append(struct_prior)
            
            # ============================================================
            # Semantic Prior: p(z_sem | y) - class-conditional
            # ============================================================
            if sem_count > 0:
                sem_prior = _build_sem_prior(
                    prior_cfg=prior_cfg,
                    sem_count=sem_count,
                    num_classes=num_classes,
                    C=C,
                    H=H, W=W,
                    device=device
                )
                level_params.append(sem_prior)
            
            level_priors.append([noise_prior, struct_prior, sem_prior])
            level_priors_params.append(level_params)
            splits.append(split)
    
    return level_priors, splits, level_priors_params


def _build_noise_prior(
    prior_cfg: DictConfig,
    noise_count: int,
    H: int, W: int,
    device: torch.device
) -> nn.Module:
    """Build unconditional noise prior p(z_noise)."""
    noise_cfg = prior_cfg.get('noise', {})
    init_strategy = noise_cfg.get('init_strategy', 'zero')
    
    # Filter out control-flow keys
    noise_params = {k: v for k, v in noise_cfg.items() if k not in ['init_strategy']}
    
    prior_type = prior_cfg.get('prior_type', 'kpmvt')
    noise_prior = create_unconditional_priors(
        K=1,
        C=noise_count, H=H, W=W,
        prior_type=prior_type,
        rank=tuple(prior_cfg.rank),
        init_strategy=init_strategy,
        **noise_params
    )[0].to(device)
    
    return noise_prior


def _build_struct_prior(
    prior_cfg: DictConfig,
    struct_count: int,
    C: int,
    H: int, W: int,
    device: torch.device
) -> nn.Module:
    """Build conditional structural prior p(z_struct^(i) | h^(i+1))."""
    prior_type = prior_cfg.get('prior_type', 'kpmvt')
    struct_cfg = prior_cfg.get('struct', {})
    struct_type = struct_cfg.get('type', 'conditional')
    
    # Filter out control-flow key
    struct_params = {k: v for k, v in struct_cfg.items() if k != 'type'}
    
    if struct_type == 'mixture':
        # Conditional mixture: multiple components with learned mixing weights
        num_modes = struct_params.get('num_modes', 4)
        struct_prior = create_conditional_mixture_prior(
            K=num_modes,
            C=struct_count, H=H, W=W,
            h_channels=C,  # Conditioning from h^(i+1)
            prior_type=prior_type,
            rank=tuple(prior_cfg.rank),
            **struct_params
        ).to(device)
        
    elif struct_type == 'conditional':
        # Standard conditional: p(z_struct | h^(i+1))
        # h_channels: latent being modeled (struct_count channels)
        # z_channels: conditioning from h^(i+1) (C channels from next level)
        struct_prior = create_conditional_prior(
            h_channels=struct_count,
            z_channels=C,
            H=H, W=W,
            prior_type=prior_type,
            rank=tuple(prior_cfg.rank),
            **struct_params
        ).to(device)
        
    else:
        raise ValueError(f"Unknown struct type: {struct_type}. Must be 'conditional' or 'mixture'")
    
    return struct_prior


def _build_sem_prior(
    prior_cfg: DictConfig,
    sem_count: int,
    num_classes: int,
    C: int,
    H: int, W: int,
    device: torch.device
) -> nn.Module:
    """Build class-conditional semantic prior p(z_sem | y)."""
    prior_type = prior_cfg.get('prior_type', 'kpmvt')
    sem_cfg = prior_cfg.get('sem', {})
    sem_type = sem_cfg.get('type', 'unconditional')

    # Filter out control-flow keys
    sem_params = {k: v for k, v in sem_cfg.items() if k not in ['type']}

    if sem_type == 'conditional_mean_shift':
        # Build a conditional semantic prior that returns a list of K distributions.
        # This enables class-dependent scoring while keeping compute_level_logits unchanged.
        sem_prior = create_conditional_prior(
            h_channels=sem_count,
            z_channels=C,
            H=H, W=W,
            prior_type=prior_type,
            rank=tuple(prior_cfg.rank),
            use_mean_shift=True,
            class_conditional_mean_shift=True,
            num_classes=num_classes,
            **sem_params,
        ).to(device)
        return sem_prior

    if sem_type != 'unconditional':
        raise ValueError(f"Unknown sem type: {sem_type}. Must be 'unconditional' or 'conditional_mean_shift'")

    sem_prior = create_class_conditional_prior(
        K=num_classes,
        C=sem_count, H=H, W=W,
        prior_type=prior_type,
        rank=tuple(prior_cfg.rank),
        **sem_params
    ).to(device)
    return sem_prior
