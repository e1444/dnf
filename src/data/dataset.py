import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra


class Dequantize(object):
    def __call__(self, tensor):
        # Adds uniform noise [0, 1/256]
        return tensor + torch.rand_like(tensor) / 256.0
    
    
def load_mnist(cfg: DictConfig):
    """
    Loads the MNIST dataset using configuration from a Hydra DictConfig.
    
    Args:
        cfg (DictConfig): Hydra configuration object, typically cfg.data.
    
    Returns:
        tuple: A tuple containing the training and test DataLoaders.
    """
    
    # Build the transformation pipeline from the config
    transform_list = []
    if cfg.dataset.transform:
        for t_cfg in cfg.dataset.transform:
            if t_cfg["type"] == "Dequantize":
                transform_list.append(Dequantize())
                continue
            
            transform_config = {'_target_': f'torchvision.transforms.{t_cfg["type"]}'}
            if 'params' in t_cfg and t_cfg.params:
                transform_config.update(t_cfg.params)
            if 'mean' in t_cfg and 'std' in t_cfg:
                transform_config['mean'] = t_cfg['mean']
                transform_config['std'] = t_cfg['std']
            transform_list.append(hydra.utils.instantiate(transform_config))

    transform = transforms.Compose(transform_list)
    
    # Create the training dataset
    train_dataset = datasets.MNIST(
        root=cfg.dataset.path,
        train=True,
        transform=transform,
        download=cfg.dataset.download
    )
    
    # Create the test dataset
    test_dataset = datasets.MNIST(
        root=cfg.dataset.path,
        train=False,
        transform=transform,
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