# import necessary libraries
import torch
from lightning import (
    Trainer,
    LightningModule, 
    LightningDataModule
)

# configuration management
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from modules.lit_ddpm import LitDDPM

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    mnist_dm: LightningDataModule = instantiate(config.data.datamodule)


    ddpm = instantiate(config.training.lit_ddpm)
    trainer: Trainer = instantiate(config.training.trainer)
    mnist_dm.setup("fit")
    trainer.fit(ddpm, mnist_dm)

    # mnist_dm.setup("test") 
    # trainer.test(autoencoder, mnist_dm)

if __name__ == "__main__":
    main()