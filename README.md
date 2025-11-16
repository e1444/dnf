# Discriminative Normalizing Flow (DNF)

This project implements a **Discriminative Normalizing Flow (DNF)** model for image classification, built with PyTorch. The primary goal of this implementation is to explore the use of normalizing flows in a discriminative setting, specifically for the MNIST dataset.

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

- Python 3.11+
- CUDA-enabled GPU (optional but recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/e1444/dnf.git
   cd dnf
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., `venv`).
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

## Experiment Tracking

This project is integrated with [Weights & Biases](https://wandb.ai) for experiment tracking. To use it:

1. **Log in to W&B**:
   ```bash
   wandb login
   ```
2. **Configure `conf/config.yaml`**:
   Update the `wandb.project` and `wandb.entity` fields with your project name and W