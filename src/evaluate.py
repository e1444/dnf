import os
import torch
import hydra
from omegaconf import DictConfig
from src.data.dataset import load_mnist
from src.models.dnf import DNFNetwork
from src.utils.evaluation import evaluate_model
import wandb

@hydra.main(config_name="config")
def main(cfg: DictConfig):
    # Initialize Weights & Biases
    wandb.init(project=cfg.wandb.project, config=cfg)

    # Load the dataset
    train_loader, test_loader = load_mnist(cfg.data)

    # Initialize the model
    model = DNFNetwork(in_channels=1, num_layers=cfg.model.num_layers, hidden_channels=cfg.model.hidden_channels)
    model.load_state_dict(torch.load(cfg.model.checkpoint_path))
    model.eval()

    # Evaluate the model
    metrics = evaluate_model(model, test_loader)

    # Log metrics to Weights & Biases
    wandb.log(metrics)

if __name__ == "__main__":
    main()