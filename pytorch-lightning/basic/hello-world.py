import os

import  torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim.optimizer import Optimizer


# Define the model
encoder = nn.Sequential(
    nn.Linear(28 * 28, 64),  # 28x28 is the size of the MNIST images,
    nn.ReLU(),
    nn.Linear(64, 3)
)

decoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 28 * 28)   
)

# define the lightning module
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # training_step defined the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1) # flatten the image
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x) # calculate the loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# define the dataset using torchvision MNIST
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(dataset) 

# init the trainer
trainer = L.Trainer(limit_train_batches=100, max_epochs=1) 
trainer.fit(LitAutoEncoder(), train_dataloaders=train_loader)

# use the model
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_images = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_images)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

