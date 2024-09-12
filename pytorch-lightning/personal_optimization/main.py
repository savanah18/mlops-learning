# import necessary libraries
import torch
from lightning import Trainer
from torch.utils.data import (
    DataLoader, 
    Dataset
)

# configuration management
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from data_utils import split_training_data
from model import LitAutoEncoder    
from train import train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    train_set: Dataset = instantiate(config.data.train)

    
    train_set, val_set  = split_training_data(
        train_set, 
        ratio=config.data.preproc.train_val_split.ratio, 
        seed=config.data.preproc.train_val_split.seed
    )
    
    train_loader: DataLoader = instantiate(config.training.train_data_loader, dataset=train_set)
    valid_loader: DataLoader = instantiate(config.training.valid_data_loader, dataset=val_set)

    autoencoder: LitAutoEncoder = instantiate(config.training.lit_auto_encoder)
    trainer: Trainer = train(autoencoder, train_loader, valid_loader, config)

    # TODO Seperate model testing from training
    # test the model
    test_set: Dataset = instantiate(config.data.test)
    test_loader: DataLoader = instantiate(config.training.test_data_loader, dataset=test_set)
    trainer.test(autoencoder, dataloaders=test_loader)


if __name__ == "__main__":
    main()