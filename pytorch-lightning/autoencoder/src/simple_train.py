# TO BE DEPRECATED
# import the necessary packages
import torch
from lightning import (
    Trainer,
    LightningModule
)
from lightning.pytorch.callbacks import DeviceStatsMonitor
from torch.utils.data import DataLoader

# configuration management
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# tensorboard logging
from lightning.pytorch.loggers import TensorBoardLogger


def train(
        model: LightningModule, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        config: DictConfig
    ) -> Trainer:
   
    trainer: Trainer = instantiate(config.training.trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer