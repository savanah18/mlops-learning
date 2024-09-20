from lightning import LightningDataModule

import torch
from torch.utils.data import (
    DataLoader, 
    Dataset, 
    random_split
)

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class MNISTDataModule(LightningDataModule):
    def __init__(
            self, 
            data_dir: str = './data',
            batch_size: int = 32,
            num_workers: int = 4,
            train_val_split_ratio: float = 0.8,
            train_val_split_seed: int = 42,
            *args,
            **kwargs
        ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split_ratio = train_val_split_ratio
        self.train_val_spit_seed = train_val_split_seed


    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        match stage:
            case "fit":            
                mnist_full: Dataset = MNIST(self.data_dir, train=True, download=True, transform=ToTensor())
                train_len = int(len(mnist_full)*self.train_val_split_ratio)
                val_len = len(mnist_full) - train_len
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full, 
                    [train_len, val_len], 
                    generator=torch.Generator().manual_seed(self.train_val_spit_seed)
                )
            case "test":
                self.mnist_test: Dataset = MNIST(self.data_dir, train=False, download=True, transform=ToTensor())
            case "predict":
                self.mnist_predict: Dataset = MNIST(self.data_dir, train=False, download=True, transform=ToTensor())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        # modify prediction if ever neededs
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
    

if __name__ == "__main__":
    MNISTDataModule()


