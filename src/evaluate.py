import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_dataset
from src.models.priors import ClassConditionalPrior
from src.utils.evaluation import evaluate, print_train_stats

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
    assert len(cfg.level_priors.priors) == cfg.model.num_levels, "Number of priors must match number of model levels"
    
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
    
    # --- DIAGNOSTIC BLOCK ---
    print("\n" + "="*50)
    print("PRIOR DIAGNOSTICS (Check for Scale Mismatch)")
    print("="*50)
    with torch.no_grad():
        for i, (noise_p, struct_p, sem_p) in enumerate(level_priors):
            print(f"\n[Level {i}]")
            
            if noise_p is not None:
                print(f"  Noise Prior ({noise_p.__class__.__name__}):")
                print(f"    tau: {noise_p.tau.item():.4f}")
                if hasattr(noise_p, '_log_det'):
                    print(f"    Internal LogDet: {noise_p._log_det.item():.4f}")
                if hasattr(noise_p, 'log_diag'):
                    print(f"    LogDiag (Mean): {noise_p.log_diag.mean().item():.4f}")
                if hasattr(noise_p, 'log_cov_ch_diag'):
                    print(f"    Ch LogDiag (Mean): {noise_p.log_cov_ch_diag.mean().item():.4f}")
                    print(f"    Sp LogDiag (Mean): {noise_p.log_cov_sp_diag.mean().item():.4f}")

            if struct_p is not None:
                print(f"  Struct Prior ({struct_p.__class__.__name__}):")
                print(f"    tau: {struct_p.tau.item():.4f}")
                # Conditional priors don't have static log_det/diag
                
            if sem_p is not None:
                print(f"  Semantic Prior ({sem_p.__class__.__name__}):")
                # ClassConditionalPrior contains a list of priors
                taus = torch.stack([p.tau for p in sem_p.priors])
                print(f"    tau (Min/Max/Mean): {taus.min().item():.4f} / {taus.max().item():.4f} / {taus.mean().item():.4f}")
                
                # Check first component for structure stats
                p0 = sem_p.priors[0]
                if hasattr(p0, '_log_det'):
                    log_dets = torch.stack([p._log_det for p in sem_p.priors])
                    print(f"    Internal LogDet (Mean): {log_dets.mean().item():.4f}")
    
    print("="*50 + "\n")
    # ------------------------
    
    print("\nRunning evaluation on Train Set...")
    train_stats = evaluate(model, train_loader, device, cfg, level_priors, splits, prefix="train_eval")

    print("\nRunning evaluation on Test Set...")
    test_stats = evaluate(model, test_loader, device, cfg, level_priors, splits, prefix="test")

    print_train_stats(epoch, train_stats, test_stats)


if __name__ == "__main__":
    main()