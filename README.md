# dnf-hydra-wandb

This project implements a Deep Neural Flow (DNF) model using PyTorch, with a focus on the MNIST dataset. It incorporates Hydra for configuration management and Weights & Biases (wandb) for experiment tracking. The project is structured to facilitate easy experimentation and model evaluation.

## Project Structure

```
dnf-hydra-wandb
├── conf
│   ├── config.yaml          # Main configuration file for Hydra
│   ├── data
│   │   └── mnist.yaml       # Configuration for MNIST dataset
│   ├── model
│   │   └── dnf_cnn.yaml     # Model architecture and hyperparameters
│   └── training
│       └── default.yaml      # Training parameters
├── src
│   ├── data
│   │   └── dataset.py       # Data loading and preprocessing
│   ├── models
│   │   └── dnf.py           # DNF model implementation
│   ├── utils
│   │   ├── evaluation.py     # Model evaluation functions
│   │   └── losses.py         # Custom loss functions
│   ├── evaluate.py           # Model evaluation script
│   └── train.py              # Main training loop
├── .gitignore                # Git ignore file
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd dnf-hydra-wandb
pip install -r requirements.txt
```

## Configuration

The project uses Hydra for configuration management. Configuration files are located in the `conf` directory. You can modify the settings in `config.yaml`, `mnist.yaml`, `dnf_cnn.yaml`, and `default.yaml` to customize the behavior of the project.

## Training

To train the model, run the following command:

```bash
python src/train.py
```

This will start the training process using the configurations specified in the YAML files.

## Evaluation

After training, you can evaluate the model's performance using:

```bash
python src/evaluate.py
```

This will load the trained model and compute evaluation metrics on the test dataset.

## Experiment Tracking

Weights & Biases (wandb) is integrated for tracking experiments. Make sure to set up your wandb account and log in before running the training script. You can monitor your experiments in real-time on the wandb dashboard.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- PyTorch for the deep learning framework.
- Hydra for configuration management.
- Weights & Biases for experiment tracking.