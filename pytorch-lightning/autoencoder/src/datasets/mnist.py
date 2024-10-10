from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataSet(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = ToTensor()

# Create other Datasets