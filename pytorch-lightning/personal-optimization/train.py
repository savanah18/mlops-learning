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

@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:

    # load the dataset
    train_set = MNIST(root="MNIST", download=config.data.train.download, train=True, transform=ToTensor())
    # create validation set
    train_set_size = int(len(train_set) * 0.8) # TODO configurable split
    val_set_size = len(train_set) - train_set_size

    # split the train set into two with seed 42
    seed = torch.Generator().manual_seed(42) #TODO seed should be configurable
    train_set, val_set = random_split(train_set, [train_set_size, val_set_size], generator=seed)
 
    train_loader= DataLoader(train_set)
    valid_loader = DataLoader(val_set)

    # define the autoencoder
    autoencoder = LitAutoEncoder(Encoder(), Decoder(), config=config)

    # train the model
    trainer = Trainer(limit_train_batches=config.training.batch_size , max_epochs=config.training.max_epochs)
    trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # test the model
    test_set = MNIST(root="MNIST", download=config.data.train.download, train=False, transform=ToTensor())
    test_loader= DataLoader(test_set)
    trainer.test(autoencoder, dataloaders=test_loader)


if __name__ == "__main__":
    main()