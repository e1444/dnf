import os
import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import load_mnist
from src.models.dnf import DNFNetwork
from src.utils.losses import deep_supervision_loss, total_loss_fn, compute_logits
from src.utils.evaluation import evaluate

def get_target_distributions(means_param, log_vars_param, num_classes, device):
    """Creates a list of target distributions."""
    return [
        torch.distributions.MultivariateNormal(
            loc=means_param[i],
            covariance_matrix=torch.diag(torch.exp(log_vars_param[i]))
        ) for i in range(num_classes)
    ]

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # Convert OmegaConf to a plain dictionary for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity, 
        config=config_dict,
        name=cfg.wandb.name
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = load_mnist(cfg.data)

    # Initialize model
    model = DNFNetwork(
        in_channels=1, 
        num_layers=cfg.model.num_layers, 
        hidden_channels=cfg.model.hidden_channels
    ).to(device)

    # Initialize trainable means and log_vars
    initial_means = torch.zeros(cfg.training.num_classes, cfg.training.features, device=device)
    for i in range(cfg.training.num_classes):
        initial_means[i, i] = cfg.training.latent_separation
    initial_means += torch.randn_like(initial_means) * cfg.training.latent_noise
    trainable_means = nn.Parameter(initial_means)

    initial_log_vars = torch.zeros(cfg.training.num_classes, cfg.training.features, device=device)
    initial_log_vars += torch.randn_like(initial_log_vars) * cfg.training.latent_noise
    trainable_log_vars = nn.Parameter(initial_log_vars)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    optimizer.add_param_group({'params': [trainable_means], 'lr': cfg.training.lr_means})
    optimizer.add_param_group({'params': [trainable_log_vars], 'lr': cfg.training.lr_vars})

    # Initialize scheduler
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    # Loss weights
    aux_layers = np.arange(cfg.model.num_layers - 1, step=cfg.training.aux_freq) + 1
    alphas = torch.tensor(np.geomspace(start=cfg.training.gamma_alpha ** (cfg.model.num_layers - 1), stop=1, num=len(aux_layers)), device=device)
    betas = torch.tensor(np.geomspace(start=cfg.training.gamma_beta ** (cfg.model.num_layers - 1), stop=1, num=len(aux_layers)), device=device)

    # Training loop
    print("Starting training...")
    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Get the current dynamic target distributions
            target_dists = get_target_distributions(trainable_means, trainable_log_vars, cfg.training.num_classes, device)

            # Forward pass
            intermediate_outputs = model(x_batch)
            aux_outputs = [intermediate_outputs[i] for i in aux_layers]
            z, log_det = intermediate_outputs[-1]
            
            aux_logits = [
                compute_logits(z, log_det, target_dists) for z, log_det in aux_outputs
            ]
            logits = compute_logits(z, log_det, target_dists)
            
            # Calculate loss
            aux_loss = deep_supervision_loss(aux_logits, y_batch, alphas, betas)
            final_loss = total_loss_fn(logits, y_batch, lambda_=cfg.training.lambda_)
            loss = aux_loss + final_loss
            
            # Regularization terms
            loss += cfg.training.r_logdet * (log_det ** 2).mean()
            loss += cfg.training.r_var * (trainable_log_vars ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([trainable_means], max_norm=1e-2)
            torch.nn.utils.clip_grad_norm_([trainable_log_vars], max_norm=1e-2)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        log_dict = {"epoch": epoch, "train_loss": avg_train_loss}

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            eval_target_dists = get_target_distributions(trainable_means, trainable_log_vars, cfg.training.num_classes, device)
            test_loss, accuracy, nll = evaluate(model, test_loader, device, cfg, eval_target_dists, alphas, betas, cfg.training.lambda_, aux_layers)
            log_dict.update({
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "test_nll": nll
            })
            print(f"Epoch [{epoch+1:02d}/{cfg.training.epochs}] | Train Loss: {avg_train_loss:.4f} | Test Acc: {accuracy:.2f}% | NLL: {nll:.4f}")

        wandb.log(log_dict)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainable_means': trainable_means,
                'trainable_log_vars': trainable_log_vars,
            }, checkpoint_path)
            wandb.save(checkpoint_path)
            
        scheduler.step()
        
    print("Training completed.")

    # Return the final accuracy for Optuna
    _, accuracy, _ = evaluate(model, test_loader, device, cfg, get_target_distributions(trainable_means, trainable_log_vars, cfg.training.num_classes, device), alphas, betas, cfg.training.lambda_, aux_layers)
    return accuracy

if __name__ == "__main__":
    train()
