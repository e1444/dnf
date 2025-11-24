import os
import hydra
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_mnist
from src.distributions.mvt import get_target_distributions
from src.utils.losses import compute_logits, nll_loss_fn, ce_loss_fn, deep_ce_loss, total_loss_fn
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
        name=cfg.wandb.name
    )
    
    # Load data
    train_loader, test_loader = load_mnist(cfg.data)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    
    with torch.no_grad():
        initial_mu = torch.zeros(cfg.training.num_classes, cfg.training.features, device=device)
        for i in range(cfg.training.num_classes):
            initial_mu[i, i] = cfg.training.latent_separation
        initial_mu += torch.randn_like(initial_mu) * cfg.training.latent_noise
        latent_mu = nn.Parameter(initial_mu)
        
        initial_v = torch.ones(cfg.training.num_classes, cfg.training.features, device=device) * torch.tensor(cfg.training.latent_v)
        initial_v += torch.randn_like(initial_v) * cfg.training.latent_noise
        # Safety: Ensure positive start
        initial_v = torch.abs(initial_v)
        latent_v = nn.Parameter(initial_v)
        
        initial_U = torch.zeros(cfg.training.num_classes, cfg.training.features, cfg.training.latent_U_size, device=device)
        latent_U = nn.Parameter(initial_U)
        
        initial_df = torch.ones(cfg.training.num_classes, device=device) * cfg.training.latent_df
        latent_df = nn.Parameter(initial_df)
        
    if cfg.training.lr == 0:
        model.requires_grad_(False)
    if cfg.training.lr_mu == 0:
        latent_mu.requires_grad_(False)
    if cfg.training.lr_v == 0:
        latent_v.requires_grad_(False)
    if cfg.training.lr_U == 0:
        latent_U.requires_grad_(False)
    if cfg.training.lr_df == 0:
        latent_df.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    optimizer.add_param_group({'params': [latent_mu], 'lr': cfg.training.lr_mu, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [latent_v], 'lr': cfg.training.lr_v, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [latent_U], 'lr': cfg.training.lr_U, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': [latent_df], 'lr': cfg.training.lr_df, 'weight_decay': 0.0})
    
    # --- Augmented Lagrangian Setup ---
    # Initialize dual variable (log_alpha)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    # Separate optimizer for alpha (dual)
    alpha_optimizer = optim.Adam([log_alpha], lr=cfg.training.lr_log_alpha)
    
    # Hyperparams
    rho = cfg.training.aug_rho
    # Default constraint: 1.0 nat per dimension if not specified
    nll_constraint = cfg.training.nll_constraint

    start_epoch = 0
    if cfg.training.resume_from_checkpoint is not None:
        print(f"Resuming training from {cfg.training.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.training.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        with torch.no_grad():
            if "latent_mu" in checkpoint:
                latent_mu.copy_(checkpoint['latent_mu'])
            if "latent_v" in checkpoint:
                latent_v.copy_(checkpoint['latent_v'])
            if "latent_U" in checkpoint:
                latent_U.copy_(checkpoint['latent_U'])
            if "latent_df" in checkpoint:
                latent_df.copy_(checkpoint['latent_df'])
        
        if not cfg.training.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]['lr'] = cfg.training.lr
            optimizer.param_groups[0]['weight_decay'] = cfg.training.weight_decay
            optimizer.param_groups[1]['lr'] = cfg.training.lr_mu
            optimizer.param_groups[2]['lr'] = cfg.training.lr_v
            optimizer.param_groups[3]['lr'] = cfg.training.lr_U
            optimizer.param_groups[4]['lr'] = cfg.training.lr_df
        start_epoch = checkpoint['epoch'] + 1
            
        # Load dual variables if available
        if 'log_alpha' in checkpoint:
            log_alpha.data.copy_(checkpoint['log_alpha'])
        if 'alpha_optimizer_state_dict' in checkpoint:
            alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
    if cfg.training.reset_freeze:
        model.set_freeze_steps(freeze=False, start=0, end=-1)
    
    if cfg.training.freeze_steps.start is not None or cfg.training.freeze_steps.end is not None:
        model.set_freeze_steps(
            freeze=True,
            start=cfg.training.freeze_steps.start,
            end=cfg.training.freeze_steps.end
        )
        
    # Initialize scheduler
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    aux_layers = np.arange(start=cfg.training.aux_freq - 1, stop=cfg.training.aux_total, step=cfg.training.aux_freq)
    betas = torch.tensor(np.geomspace(start=cfg.training.gamma_beta ** len(aux_layers), stop=1, num=len(aux_layers)), device=device)

    # Training loop
    print("Starting training...")
    total_epochs = start_epoch + cfg.training.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_ce = 0.0
        total_alpha = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # --- Primal Step ---
            optimizer.zero_grad()

            # Forward pass
            intermediate_outputs = model(x_batch)
            # Flatten outputs
            intermediate_outputs = [(z.view(z.size(0), -1), log_det) for z, log_det in intermediate_outputs]
            aux_outputs = [intermediate_outputs[i] for i in aux_layers]
            z, log_det = intermediate_outputs[-1]
            
            # Get the current dynamic target distributions
            # Note: We allow gradients to flow through latent params here
            target_dists = get_target_distributions(latent_mu, latent_v, latent_U, latent_df, cfg.training.num_classes, cfg.training.latent_v_eps, device=device)
            
            # Compute logits
            aux_logits = [
                compute_logits(z, log_det, target_dists) for z, log_det in aux_outputs
            ]
            logits = compute_logits(z, log_det, target_dists)
            
            # 1. Task Loss (CE + Aux)
            aux_ce_loss = deep_ce_loss(aux_logits, y_batch, betas, label_smoothing=cfg.training.label_smoothing)
            ce_loss = ce_loss_fn(logits, y_batch, label_smoothing=cfg.training.label_smoothing)
            task_loss = (1 - cfg.training.lambda_) * aux_ce_loss + cfg.training.lambda_ * ce_loss
            
            # 2. NLL Loss (The Constraint)
            nll_loss = nll_loss_fn(logits, y_batch)
            
            # 3. Regularization terms (Jacobian smoothness)
            log_dets = [log_det for _, log_det in intermediate_outputs]
            for i in reversed(range(1, len(log_dets))):
                log_dets[i] = log_dets[i] - log_dets[i - 1]
            reg_loss = cfg.training.r_logdet * (torch.stack(log_dets) ** 2).mean()
            
            # Primal Objective
            primal_loss = task_loss
            primal_loss += reg_loss
            
            alpha = F.softplus(log_alpha)
            nll_violation = nll_loss - nll_constraint
            violation_pos = F.relu(nll_violation)
            
            if cfg.training.use_al:
                primal_loss += alpha * nll_violation 
                primal_loss += 0.5 * rho * violation_pos.pow(2)

            primal_loss.backward()
            
            # Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradclip)
            if latent_mu.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_mu], max_norm=cfg.training.gradclip_mu)
            if latent_v.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_v], max_norm=cfg.training.gradclip_v)
            if latent_U.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_U], max_norm=cfg.training.gradclip_U)
            if latent_df.requires_grad:
                torch.nn.utils.clip_grad_norm_([latent_df], max_norm=cfg.training.gradclip_df)
            optimizer.step()
            
            # Clamping
            if latent_v.requires_grad:
                with torch.no_grad():
                    latent_v.data.clamp_(min=1e-5)
            if latent_df.requires_grad:
                with torch.no_grad():
                    latent_df.data.clamp_(min=2.0 + 1e-4)  # df > 2 for finite variance

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
            eval_target_dists = get_target_distributions(latent_mu, latent_v, latent_U, latent_df, cfg.training.num_classes, cfg.training.latent_v_eps, device=device)
            test_loss, test_accuracy, test_nll = evaluate(model, test_loader, device, cfg, eval_target_dists, betas, cfg.training.lambda_, aux_layers)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_nll": test_nll
            })
            
            train_loss, train_accuracy, train_nll = evaluate(model, train_loader, device, cfg, eval_target_dists, betas, cfg.training.lambda_, aux_layers)
            log_dict.update({
                "train_eval_loss": train_loss,
                "train_eval_accuracy": train_accuracy,
                "train_eval_nll": train_nll
            })
            
            print(f"Epoch [{epoch+1:02d}/{total_epochs}] | Loss: {avg_train_loss:.4f} | Acc (Tr/Te): {train_accuracy:.2f}%/{test_accuracy:.2f}% | NLL: {avg_nll:.2f} (Target {nll_constraint:.1f}) | Alpha: {avg_alpha:.4f}")


        wandb.log(log_dict)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'latent_mu': latent_mu.detach().cpu().clone(),
                'latent_v': latent_v.detach().cpu().clone(),
                'latent_U': latent_U.detach().cpu().clone(),
                'latent_df': latent_df.detach().cpu().clone(),
                'log_alpha': log_alpha.detach().cpu().clone(),
                'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
            }, checkpoint_path)
            wandb.save(checkpoint_path)
            
        if scheduler is not None:
            # Check if it's ReduceLROnPlateau (requires a metric)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_train_loss)
            else:
                # For other schedulers (StepLR, Cosine, etc.)
                scheduler.step()
        
    print("Training completed.")

    # Return the final accuracy for Optuna
    _, accuracy, _ = evaluate(model, test_loader, device, cfg, get_target_distributions(latent_mu, latent_v, latent_U, latent_df, cfg.training.num_classes, cfg.training.latent_v_eps, device=device), betas, cfg.training.lambda_, aux_layers)
    return accuracy

if __name__ == "__main__":
    train()
