# import necessary libraries
import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F

# configuration management
import hydra
from omegaconf import DictConfig

from modules import Encoder, Decoder

# einiops
from einops import rearrange


# create a lightning autoencoder
class LitAutoEncoder(LightningModule):
    def __init__(self,
        encoder: Encoder, 
        decoder: Decoder, 
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()
        self.save_hyperparameters() #save the hyperparameters
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: Tensor) -> Tensor:
        # x = x.view(x.size(0), -1) # flatten the image
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # return x_hat
        x = rearrange(x, 'b c h w -> b (c h w)')
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = rearrange(x_hat, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return x_hat

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
        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        return optimizer