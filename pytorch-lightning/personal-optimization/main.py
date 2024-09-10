# import necessary libraries
import torch
from lightning import Trainer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

# configuration management
import hydra
from omegaconf import DictConfig

from modules import Encoder, Decoder
from model import LitAutoEncoder    
from train import train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    # TODO: Seperata Data Preparation from Model Training
    train_set = MNIST(root="MNIST", download=config.data.train.download, train=True, transform=ToTensor())
    train_set_size = int(len(train_set) * 0.8) # TODO configurable split
    val_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42) #TODO seed should be configurable
    train_set, val_set = random_split(train_set, [train_set_size, val_set_size], generator=seed)
 
    train_loader= DataLoader(
        train_set, 
        num_workers=config.data.train.dataloader.num_workers, 
        batch_size=config.training.limit_train_batches
    )
    valid_loader = DataLoader(
        val_set,
        num_workers=config.data.valid.dataloader.num_workers,
        batch_size=config.training.limit_val_batches
    )

    # define the autoencoder
    autoencoder = LitAutoEncoder(Encoder(), Decoder(), config=config)
    trainer: Trainer = train(autoencoder, train_loader, valid_loader, config)

    # TODO Seperate model testing from training
    # test the model
    test_set = MNIST(root="MNIST", download=config.data.train.download, train=False, transform=ToTensor())
    test_loader= DataLoader(
        test_set,
        num_workers=config.data.test.dataloader.num_workers,
        batch_size=config.training.limit_test_batches
    )
    trainer.test(autoencoder, dataloaders=test_loader)


if __name__ == "__main__":
    main()