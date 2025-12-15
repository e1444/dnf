import os
import hydra
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
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg.training))
    assert cfg.training.resume_from_checkpoint is not None, "Evaluation requires a trained checkpoint to resume from" 
    
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
    splits = []
    level_priors_params = nn.ModuleList()
    
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
                    H=H, W=W,
                    rank=prior_cfg.rank
                ).to(device)
                level_params.append(struct_prior)
            
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

    print(f"Loaded checkpoint from {cfg.training.resume_from_checkpoint}")
    checkpoint = torch.load(cfg.training.resume_from_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
    
    epoch = checkpoint['epoch'] + 1
    
    print("\nRunning evaluation on Train Set...")
    train_stats = evaluate(model, train_loader, device, cfg, level_priors, splits, prefix="train_eval")

    print("\nRunning evaluation on Test Set...")
    test_stats = evaluate(model, test_loader, device, cfg, level_priors, splits, prefix="test")

    print("\n" + "="*60)
    print(f"Evaluation Results (Epoch {epoch})")
    print("="*60)
    print(f"Loss: {train_stats['train_eval_loss']:.4f} (Train) | {test_stats['test_loss']:.4f} (Test)")
    print(f"Accuracy: {train_stats['train_eval_accuracy']:.2f}% (Train) | {test_stats['test_accuracy']:.2f}% (Test)")
    print(f"NLL: {train_stats['train_eval_nll']:.2f} (Train) | {test_stats['test_nll']:.2f} (Test)")
    print("-" * 60)

    split_names = ["Noise", "Structure", "Semantics"]
    
    def print_split_matrix(stats, title):
        avg_logit_split = stats[f'{title.lower()}_logit_split']  # (3, L)
        level_labels = [f"Level {i}" for i in range(avg_logit_split.shape[1])]
        
        print(f"\n{title} Logit Split Contributions (Avg over samples):")
        header = " " * 12 + "".join(f"{lvl:>12}" for lvl in level_labels)
        print(header)
        print("-" * len(header))
        
        for i, name in enumerate(split_names):
            row = "".join(f"{avg_logit_split[i, j]:>12.4f}" for j in range(avg_logit_split.shape[1]))
            print(f"{name:<12} {row}")
        print("")

    print_split_matrix(test_stats, "Test")
    print_split_matrix(train_stats, "Train_Eval")
    print("="*60)


if __name__ == "__main__":
    main()