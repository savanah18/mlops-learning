# import necessary libraries
import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# configuration management
import hydra
from omegaconf import DictConfig

# einiops
from einops import rearrange
# from diffusion import NoiseScheduler
from .diffusion import NoiseScheduler
from .unet import UNet

class LitDDPM(LightningModule):
    def __init__(
            self,
            scheduler,
            model,
            *args,
            **kwargs
        ) -> None:
        super().__init__()
        self.scheduler: NoiseScheduler = scheduler
        self.model = model
        self.model_args = kwargs['model_args']
        self.training_args = kwargs['training_args']
        self.save_hyperparameters()

    
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        im, _ = batch
        im = im.float()
        
        # Sample random noise
        noise = torch.randn_like(im)
        
        # Sample timestep
        t = torch.randint(0, self.model_args['num_timesteps'], (im.shape[0],)).to(im.device)
        
        # Add noise to images according to timestep
        noisy_im = self.scheduler.forward_process(im, noise, t)
        noise_pred = self.model(noisy_im, t)

        #loss = F.mse_loss(noise_pred, noise)
        loss = torch.nn.MSELoss()(noise_pred, noise)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # x, _ = batch
        # noise = torch.rand_like(x)
        # t = torch.randint(0, self.model_args['num_timesteps'], (x.shape[0],)).to(x.device)
        # x_t = self.scheduler.forward_process(x, noise, t)
        # noise_pred = self.model(x_t, t)

        # #val_loss = F.mse_loss(noise_pred, noise)
        # val_loss = torch.nn.MSELoss()(noise_pred, noise)
        # self.log('val_loss', val_loss)
        # return val_loss
        pass

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
        # TODO

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_args['learning_rate'])
        #scheduler= LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.training_args['warmup_epochs'], max_epochs=self.training_args['max_epochs'])   
        #return [optimizer], [scheduler]
        return optimizer
            