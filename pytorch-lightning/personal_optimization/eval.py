import torch

# dataset and dataloaders
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

# model
from model import LitAutoEncoder

# configuration management
import hydra
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name="config")
def eval(config):
    eval_config = config.eval
    print(eval_config)
    print(f"Loading model from checkpoint {eval_config.checkpoint.path}")

    # Get current working directory absolute path
    cwd = hydra.utils.get_original_cwd()

    model: LitAutoEncoder = LitAutoEncoder.load_from_checkpoint(Path(cwd)/eval_config.checkpoint.path)

    # load the dataset
    test_set: Dataset = instantiate(config.data.test)
    test_loader: DataLoader = instantiate(config.eval.test_data_loader, dataset=test_set)

    # test the model
    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(model.device)
    x_hat = model(x)
    print(x.size(), x_hat.size())


if __name__ == "__main__":
    eval()