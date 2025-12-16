import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import numpy as np
import PIL.Image


class Dequantize(object):
    def __call__(self, x):
        # Adds uniform noise [0, 1/256]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, PIL.Image.Image):
            x = torch.from_numpy(np.array(x))
            
        # 2. FIX: Add Channel Dimension if missing (H, W) -> (1, H, W)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # Handle RGB (H, W, C) -> (C, H, W)
        elif x.ndim == 3 and x.shape[2] == 3:
            x = x.permute(2, 0, 1)
            
        x = x.float()
        
        x = (x + torch.rand_like(x)) / 256.0
        return torch.clamp(x, 0.0, 1.0)
    

class AddAWGN(object):
    def __init__(self, snr_db=9.5):
        self.snr_db = snr_db

    def __call__(self, x):
        # x is [C, H, W] in [0, 1]
        # Calculate signal power
        signal_power = torch.mean(x ** 2)
        
        # Avoid division by zero for empty images
        if signal_power == 0:
            return x
            
        # Calculate noise power required
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        
        # Add noise
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0.0, 1.0)
    
    
def load_dataset(cfg: DictConfig):
    """
    Loads a dataset (MNIST or CIFAR10) using configuration from a Hydra DictConfig.
    """
    
    # Build the transformation pipeline from the config
    train_transform_list = []
    test_transform_list = []
    
    if cfg.dataset.train_transform:
        for t_cfg in cfg.dataset.train_transform:
            if t_cfg["type"] == "Dequantize":
                train_transform_list.append(Dequantize())
                continue
            elif t_cfg["type"] == "AddAWGN":
                snr_db = t_cfg.params.get("snr_db", 9.5)
                train_transform_list.append(AddAWGN(snr_db=snr_db))
                continue
            
            transform_config = {'_target_': f'torchvision.transforms.{t_cfg["type"]}'}
            if 'params' in t_cfg and t_cfg.params:
                transform_config.update(t_cfg.params)
            if 'mean' in t_cfg and 'std' in t_cfg:
                transform_config['mean'] = t_cfg['mean']
                transform_config['std'] = t_cfg['std']
            train_transform_list.append(hydra.utils.instantiate(transform_config))
    
    if cfg.dataset.test_transform:
        for t_cfg in cfg.dataset.test_transform:
            if t_cfg["type"] == "Dequantize":
                test_transform_list.append(Dequantize())
                continue
            elif t_cfg["type"] == "AddAWGN":
                snr_db = t_cfg.params.get("snr_db", 9.5)
                test_transform_list.append(AddAWGN(snr_db=snr_db))
                continue
            
            transform_config = {'_target_': f'torchvision.transforms.{t_cfg["type"]}'}
            if 'params' in t_cfg and t_cfg.params:
                transform_config.update(t_cfg.params)
            if 'mean' in t_cfg and 'std' in t_cfg:
                transform_config['mean'] = t_cfg['mean']
                transform_config['std'] = t_cfg['std']
            test_transform_list.append(hydra.utils.instantiate(transform_config))
            
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
    
    # Select Dataset Class
    if cfg.dataset.name == "MNIST":
        dataset_cls = datasets.MNIST
    elif cfg.dataset.name == "CIFAR10":
        dataset_cls = datasets.CIFAR10
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported.")

    # Create the training dataset
    train_dataset = dataset_cls(
        root=cfg.dataset.path,
        train=True,
        transform=train_transform,
        download=cfg.dataset.download
    )
    
    # Create the test dataset
    test_dataset = dataset_cls(
        root=cfg.dataset.path,
        train=False,
        transform=test_transform,
        download=cfg.dataset.download
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers
    )
    
    return train_loader, test_loader