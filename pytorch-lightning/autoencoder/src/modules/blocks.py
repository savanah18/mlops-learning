# import necessary libraries
import torch
from torch import nn
import lightning as L
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim.optimizer import Optimizer

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 64),  # 28x28 is the size of the MNIST images,
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)   
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.l1(x)
    
