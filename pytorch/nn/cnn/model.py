from torch.nn import (
    Conv2d, 
    MaxPool2d, 
    Linear, 
    Flatten, 
    Module, 
    Sequential,
    ReLU
)
import torch
from einops import rearrange


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Simple feature map block
        self.feature_map = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3), #B,32, 28-3+1,28-3+1 
            ReLU(), 
            MaxPool2d(kernel_size=2), #B,32, 13, 13
        )
        # Simple Classifier block
        self.classifier = Linear(in_features=32*13*13, out_features=10)

    def forward(self, x):
        x = self.feature_map(x)
        x = rearrange(x, 'b c h w -> b (c h w)') #flatten
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    cnn = CNN()
    print(cnn(x).shape)