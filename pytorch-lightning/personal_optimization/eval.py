from lightning import LightningDataModule

# dataset and dataloaders
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

# model
from model import LitAutoEncoder

# configuration management
import hydra
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name="config")
def simple_eval(config):
    eval_config = config.eval
    print(eval_config)
    print(f"Loading model from checkpoint {eval_config.checkpoint.path}")

    # Get current working directory absolute path
    cwd = hydra.utils.get_original_cwd()
    model: LitAutoEncoder = LitAutoEncoder.load_from_checkpoint(Path(cwd)/eval_config.checkpoint.path)
    mnist_dm: LightningDataModule = instantiate(config.data.data_module)
    mnist_dm.setup("predict")
    predict_dataloader: DataLoader = mnist_dm.predict_dataloader()

    # test the model
    model.eval()
    x, _ = next(iter(predict_dataloader))
    with torch.no_grad():
        x = x.to(model.device)
        x_hat = model(x)
        print(x.size(), x_hat.size())


if __name__ == "__main__":
    simple_eval()