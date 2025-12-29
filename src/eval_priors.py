import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_dataset
from src.models.priors import ClassConditionalPrior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def eval_priors(cfg: DictConfig):
    print(f"Device: {device}")
    
    # Load data (just to get shapes)
    train_loader, _ = load_dataset(cfg.data)
    input_shape = next(iter(train_loader))[0].shape[1:]  # (C, H, W)
    K = cfg.data.dataset.num_classes
    
    if cfg.model._target_ == "src.models.glow.DGLOWNetwork":
        from src.models.glow import DGLOWNetwork
        output_shapes = DGLOWNetwork.output_shapes(input_shape, cfg.model.num_levels)
    else:
        raise NotImplementedError(f"Model {cfg.model._target_} not supported.")
    
    assert len(cfg.level_priors.priors) == cfg.model.num_levels, "Number of priors must match number of model levels"
    
    level_priors_params = nn.ModuleList()
    level_priors_struct = [] # To store the structure for printing
    
    print("\nInitializing Priors...")
    
    with torch.no_grad():
        for i, prior_cfg in enumerate(cfg.level_priors.priors.values()):
            C, H, W = output_shapes[i]
            split = prior_cfg.split
            noise_count, struct_count, sem_count = split
            
            print(f"\nLevel {i}: Shape ({C}, {H}, {W}), Split {split}")
            
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
                    # Create a copy to avoid modifying the original config in the loop if reused
                    cond_cls_cfg = prior_cfg.conditional_cls.copy()
                    del cond_cls_cfg.num_components
                    
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
                        cond_cls_cfg,
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
                
            level_priors_params.append(level_params)
            level_priors_struct.append({
                "noise": noise_prior,
                "struct": struct_prior,
                "sem": sem_prior
            })

    # Load from checkpoint if specified
    if cfg.training.ckpt is not None:
        print(f"\nLoading checkpoint from {cfg.training.ckpt}")
        checkpoint = torch.load(cfg.training.ckpt, map_location=device)
        
        if 'prior_state_dict' in checkpoint:
            level_priors_params.load_state_dict(checkpoint['prior_state_dict'])
            print("Prior state loaded successfully.")
        else:
            print("Warning: 'prior_state_dict' not found in checkpoint.")
    else:
        print("\nNo checkpoint specified. Using initialized values.")

    print("\n" + "="*50)
    print("PRIOR STATISTICS")
    print("="*50)
    
    for i, priors in enumerate(level_priors_struct):
        print(f"\nLevel {i}:")
        if priors["noise"]:
            print(f"  Noise Prior: {priors['noise']}")
        if priors["struct"]:
            print(f"  Struct Prior: {priors['struct']}")
        if priors["sem"]:
            print(f"  Semantic Prior: {priors['sem']}")
            # Optionally print details of the first component to see tau/df
            if hasattr(priors["sem"], "priors") and len(priors["sem"].priors) > 0:
                 print(f"    Component 0: {priors['sem'].priors[0]}")

if __name__ == "__main__":
    eval_priors()
