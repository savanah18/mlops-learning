# import necessary libraries
import os

import torch
from torch import nn
from lightning import LightningModule, Trainer
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from modules import Encoder, Decoder



# create a lightning autoencoder
class LitAutoEncoder(LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1) # flatten the image
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_loss = F.mse_loss(x_hat, x)
        self.log('train_loss', train_loss)
        return train_loss
    
    # define validation step
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        return val_loss
    
    # define test step
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log('test_loss', test_loss)
        return test_loss
    
    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == "__main__":
    # load the dataset
    train_set = MNIST(root="MNIST", download=True, train=True, transform=ToTensor())
    # create validation set
    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size

    # split the train set into two with seed 42
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_set, [train_set_size, val_set_size], generator=seed)
 
    train_loader= DataLoader(train_set)
    valid_loader = DataLoader(val_set)

    # define the autoencoder
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train the model
    trainer = Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # test the model
    test_set = MNIST(root="MNIST", download=True, train=False, transform=ToTensor())
    test_loader= DataLoader(test_set)
    trainer.test(autoencoder, dataloaders=test_loader)
