# TODO Data Loaders
from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data import random_split


def split_training_data(
        dataset: Dataset, 
        ratio=0.8, 
        seed=42
    ) -> tuple[Dataset, Dataset]:
    train_set_size = int(len(dataset) * ratio) # TODO configurable split
    val_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(seed) #TODO seed should be configurable
    return random_split(dataset, [train_set_size, val_set_size], generator=seed)