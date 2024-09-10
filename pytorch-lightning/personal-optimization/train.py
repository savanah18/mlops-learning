# import the necessary packages
import torch
from lightning import (
    Trainer,
    LightningModule
)
from torch.utils.data import DataLoader

# configuration management
import hydra
from omegaconf import DictConfig

# tensorboard logging
from lightning.pytorch.loggers import TensorBoardLogger


def train(
    model: LightningModule, 
    train_loader: DataLoader, 
    valid_loader: DataLoader,
    config: DictConfig) -> None:

    logger = TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=config.logging.train.name
    )
   
    trainer = Trainer(
            limit_train_batches=config.training.limit_train_batches ,
            limit_val_batches=config.training.limit_val_batches,
            limit_test_batches=config.training.limit_test_batches,
            max_epochs=config.training.max_epochs,
            logger=logger,
            fast_dev_run=config.training.fast_dev_run,
            num_sanity_val_steps=config.training.num_sanity_val_steps
        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer