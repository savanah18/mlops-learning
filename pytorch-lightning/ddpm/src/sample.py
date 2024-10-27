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

# models
from modules.lit_ddpm import LitDDPM

# others
from torchvision.utils import make_grid
from tqdm import tqdm
import torchvision
import os
from pathlib import Path

cwd = Path(os.getcwd())

def sample(model, scheduler, sampling_config):
    model.eval()
    device = sampling_config.device
    B, C, H, W = sampling_config.num_samples, sampling_config.model_config.im_channels, sampling_config.model_config.im_size.h, sampling_config.model_config.im_size.w
    xt = torch.randn((B, C, H, W)).to(device)

    for i in tqdm(reversed(range(sampling_config.scheduler_config.num_timesteps))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        xt, x0_pred = scheduler.reverse_process(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1.) / 2.
        grid = make_grid(ims, nrow=sampling_config.visualization.num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid)

        save_dir = os.path.join('samples', sampling_config.visualization.task_name)
        if not os.path.exists(save_dir):
            (cwd/save_dir).mkdir( parents=True, exist_ok=True )
        
        save_img = os.path.join(save_dir, 'x0_{}.png'.format(i))
        img.save(save_img)
        img.close()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    # load lightning model from checkpoint
    ddpm = LitDDPM.load_from_checkpoint(config.sample.sampling.model_ckpt)
    with torch.no_grad():
        sample(ddpm.model, ddpm.scheduler, config.sample.sampling)


if __name__ == "__main__":
    main()