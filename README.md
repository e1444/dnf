# Discriminative Normalizing Flow (DNF)

This project implements a **Discriminative Normalizing Flow (DNF)** model for image classification, built with PyTorch. The primary goal of this implementation is to explore the use of normalizing flows in a discriminative setting, specifically for the MNIST dataset. The project is structured for clarity, reproducibility, and ease of experimentation, leveraging Hydra for configuration management and Weights & Biases for comprehensive experiment tracking.

## Key Features

- **Discriminative Normalizing Flow**: Implements a DNF model, which combines the generative power of normalizing flows with a discriminative training objective.
- **Deep Supervision**: Utilizes deep supervision to improve training stability and performance by applying auxiliary losses at intermediate layers of the network.
- **Dynamic Target Distributions**: The target distributions for the latent space are treated as trainable parameters, allowing the model to learn an optimal latent representation.
- **Hydra for Configuration**: All model, training, and data parameters are managed through YAML configuration files, enabling flexible and organized experimentation.
- **Weights & Biases Integration**: Seamlessly logs training progress, evaluation metrics, and model checkpoints to Weights & Biases for easy monitoring and comparison of experiments.

## Project Structure

The project is organized as follows:

```
.
├── conf/                  # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── data/              # Data-related configurations
│   ├── model/             # Model architecture configurations
│   └── training/          # Training loop configurations
├── notebooks/             # Jupyter notebooks for analysis and visualization
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model definitions (DNF, coupling layers, etc.)
│   ├── utils/             # Utility functions (losses, evaluation)
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional but recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/e1444/dnf.git
   cd dnf
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., `venv` or `conda`).
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## How to Run

### Training

The model can be trained using the `train.py` script. Hydra allows for easy modification of parameters from the command line.

**Default Training:**
To run training with the default configuration (`conf/config.yaml`):
```bash
python src/train.py
```

**Customizing Parameters:**
You can override any configuration parameter from the command line. For example, to change the learning rate and number of epochs:
```bash
python src/train.py training.lr=0.0005 training.epochs=50
```

**Multi-Run Sweeps:**
Hydra's sweeper plugins can be used to run hyperparameter sweeps. For example, to sweep over different learning rates:
```bash
python src/train.py --multirun training.lr=0.001,0.0005,0.0001
```

### Evaluation

To evaluate a trained model, you need to provide the path to a model checkpoint.
```bash
python src/evaluate.py model.checkpoint_path=/path/to/your/checkpoint.pth
```
Evaluation results, including accuracy and Negative Log-Likelihood (NLL), will be logged to Weights & Biases.

## The DNF Model

The core of the project is the `DNFNetwork`, which is a type of normalizing flow composed of several building blocks:

- **Squeeze Operation**: Reshapes the input tensor to increase channel depth while reducing spatial dimensions.
- **Activation Normalization**: A data-dependent normalization layer that stabilizes training.
- **Invertible 1x1 Convolution**: Allows for permutation of channels, improving the expressive power of the flow.
- **Affine Coupling Layers**: The main transformation block, where half of the input dimensions are transformed based on the other half. The coupling network uses a `CNNCouplingLayer` with residual blocks.

The model is trained with a combination of a discriminative loss and an auxiliary loss from deep supervision, encouraging a well-structured latent space.

## Experiment Tracking

This project is integrated with [Weights & Biases](https://wandb.ai) for experiment tracking. To use it:

1. **Log in to W&B**:
   ```bash
   wandb login
   ```
2. **Configure `conf/config.yaml`**:
   Update the `wandb.project` and `wandb.entity` fields with your project name and W