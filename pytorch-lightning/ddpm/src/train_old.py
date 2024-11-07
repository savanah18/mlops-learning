import torch
from einops import rearrange

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_min_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward_process(self, x_0, noise, t):
        original_shape = x_0.shape #BxCxHxW
        batch_size = x_0.shape[0]
        
        assert t.shape[0] == batch_size
        sqrt_alpha_bar = rearrange(self.sqrt_alpha_bar.to(x_0.device)[t], 't -> t 1 1 1')
        sqrt_one_min_alpha_bar = rearrange(self.sqrt_one_min_alpha_bar.to(x_0.device)[t], 't -> t 1 1 1')
        
        return sqrt_alpha_bar * x_0 + sqrt_one_min_alpha_bar*noise
    
    def reverse_process(self, x_t, noise_pred, t):
        #print("debug", x_t.shape, noise_pred.shape, t.shape)
        x_0_bar = (x_t - self.sqrt_one_min_alpha_bar.to(x_t.device)[t]*noise_pred) / self.sqrt_alpha_bar.to(x_t.device)[t]
        x_0_bar = torch.clamp(x_0_bar, -1, 1)

        mu_theta = (x_t - (self.betas.to(x_t.device)[t]*noise_pred/self.sqrt_one_min_alpha_bar.to(x_t.device)[t]))/torch.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mu_theta, x_0_bar
        else:
            sigma = torch.sqrt(self.betas.to(x_t.device)[t]*(1 - self.alpha_bar.to(x_t.device)[t-1])/(1- self.alpha_bar.to(x_t.device)[t]))
            z = torch.randn_like(x_t).to(x_t.device)
            return mu_theta + sigma*z, x_0_bar
        
def get_time_embedding(time_steps, t_emb_dim):
    factor = 1e4 ** ((
        torch.arange(
            start = 0,
            end = t_emb_dim // 2,
            device = time_steps.device,
            ) / (t_emb_dim // 2)
        )
    )
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor 
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)
    return t_emb

from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.residual_input_conv =  nn.Conv2d(in_channels, out_channels, 1)
        self.down_sample = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        # Renet Block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None] #TXBX1X1
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention Block
        B, C, H, W = out.shape
        in_attn = rearrange(out, 'b c h w -> b c (h w)')
        in_attn = self.attention_norm(in_attn)
        in_attn = rearrange(in_attn, 'b c (h w) -> b (h w) c', h=H, w=W)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = rearrange(out_attn, 'b (h w) c -> b c h w', h=H, w=W)
        out = out + out_attn

        ## loop above if multiple layer
        out = self.down_sample(out)
        return out
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            )
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            )
        ])

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 1)
        ])

    def forward(self, x, t_emb):
        out = x
        resnet_input = out

        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        B, C, H, W = out.shape
        in_attn = rearrange(out, 'b c h w -> b c (h w)')
        in_attn = self.attention_norm(in_attn)
        in_attn = rearrange(in_attn, 'b c (h w) -> b (h w) c', h=H, w=W)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = rearrange(out_attn, 'b (h w) c -> b c h w', h=H, w=W)
        out = out + out_attn

        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.up_sample = nn.ConvTranspose2d(in_channels//2, in_channels//2, 4, 2, 1) if up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample(x)
        x = torch.cat([x, out_down], dim=1)

        # Renet Block
        # Loop below if multiple layers
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention Block
        B, C, H, W = out.shape
        in_attn = rearrange(out, 'b c h w -> b c (h w)')
        in_attn = self.attention_norm(in_attn)
        in_attn = rearrange(in_attn, 'b c (h w) -> b (h w) c', h=H, w=W)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = rearrange(out_attn, 'b (h w) c -> b c h w', h=H, w=W)
        out = out + out_attn

        return out
    
class Unet(nn.Module):
    def __init__(self, im_channels, **kwargs):
        super().__init__()
        self.im_channels = im_channels
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.t_emb_dim = 128
        self.down_sample = [True, True, False]

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], 3, 1, 1)

        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, self.down_sample[i],  num_heads=4)
            )
        
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim, num_heads=4)
            )

        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlock(self.down_channels[i]*2, self.down_channels[i-1] if i!=0 else 16, self.t_emb_dim, self.down_sample[i], num_heads=4)
            )

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, 3, 1, 1)

    def forward(self, x, t):
        assert x.shape[1] == self.im_channels
        out = self.conv_in(x)
        t_emb = self.t_proj(get_time_embedding(t, self.t_emb_dim))

        down_outs = []
        #print("down")
        for down in self.downs:
            #print(out.shape)
            down_outs.append(out)
            out = down(out, t_emb)

        #print("mid")
        for mid in self.mids:
            #print(out.shape)
            out = mid(out, t_emb)

        #print("up")
        for up in self.ups:
            down_out = down_outs.pop()
            #print(out.shape, down_out.shape)
            out = up(out, down_out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out
    
import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, split, im_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor


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
from modules.unet_cand import UNet


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    mnist_dm: LightningDataModule = instantiate(config.data.datamodule)


    scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                     beta_start=0.0001,
                                     beta_end=0.02)

    model = UNet(
        **config.training.lit_ddpm.model
    )

    ddpm = LitDDPM(
        scheduler=scheduler,
        model=model,
        model_args=config.training.lit_ddpm.model_args,
        training_args=config.training.lit_ddpm.training_args
    )
    trainer: Trainer = instantiate(config.training.trainer)
    mnist_dm.setup("fit")
    trainer.fit(ddpm, mnist_dm)

if __name__ == "__main__":
    main()
