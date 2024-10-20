import torch
from torch import (
    nn,
    Tensor
)

from einops import (
    rearrange,
    repeat
)


# import SinusoidalPositionalEncodingBlock from src.modules.blocks
from blocks import SinusoidalPositionalEncodingBlock

class TimeProjectionBlock(nn.Module):
    def __init__(self, t_emb_dim: int) -> None:
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
    
    def forward(self, t: Tensor) -> Tensor:
        return self.t_proj(t)
    
class ResnetConvBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1, 
            groups: int = 8,
        ):
        super().__init__()
        self.resnet_conv = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride=stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet_conv(x)
    
class ResnetAttentionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 8,
            groups: int = 8,
        )-> None:
        super().__init__()
        self.attention_norm = nn.GroupNorm(groups, in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim = in_channels,
            num_heads = num_heads,
            batch_first=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.attention_norm(x)
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=H, w=W)
        x, _ = self.attention(x, x, x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
    
class UNetDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            t_emb_dim: int,
            down_sample: bool = True,
            num_heads: int = 4,
        ) -> None:
        super().__init__() 
        self.resnet_conv_1 = ResnetConvBlock(in_channels, out_channels)
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_2 = ResnetConvBlock(out_channels, out_channels)
        self.resnet_attention = ResnetAttentionBlock(out_channels, num_heads)

        self.residual_input_conv =  nn.Conv2d(in_channels, out_channels, 1)
        self.down_sample = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        out = x
        resnet_input = x
        out = self.resnet_conv_1(out)
        # add time embedding
        out += rearrange(self.t_emb_layers(t_emb), 'b c -> b c 1 1') #??
        out = self.resnet_conv_2(out)        
        # add residual connection
        out += self.residual_input_conv(resnet_input)
        # apply attention
        out_attn = self.resnet_attention(out)
        # apply second skip connection
        out += out_attn
        # downsample if necessary
        out = self.down_sample(out)
        return out
    
class UNetMidBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            t_emb_dim: int,
            num_heads: int = 4,
        ) -> None:
        super().__init__()
        self.resnet_conv_1 = nn.ModuleList([
            ResnetConvBlock(in_channels, out_channels),
            ResnetConvBlock(out_channels, out_channels)
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
        self.resnet_conv_2 = nn.ModuleList([
            ResnetConvBlock(out_channels, out_channels),
            ResnetConvBlock(out_channels, out_channels)
        ])
        self.resnet_attention = ResnetAttentionBlock(out_channels, num_heads)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 1)
        ])

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        out = x
        resnet_input = x

        out = self.resnet_conv_1[0](out)
        out += rearrange(self.t_emb_layers[0](t_emb), 'b c -> b c 1 1') #??
        out = self.resnet_conv_2[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        out_attn = self.resnet_attention(out)
        out = out + out_attn

        resnet_input = out
        out = self.resnet_conv_1[1](out)
        out += rearrange(self.t_emb_layers[1](t_emb), 'b c -> b c 1 1')
        out = self.resnet_conv_2[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out
    
class UNetUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            t_emb_dim: int,
            up_sample: bool = True,
            num_heads: int = 4,
        ) -> None:
        super().__init__()
        self.resnet_conv_1 = ResnetConvBlock(in_channels, out_channels)
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_2 = ResnetConvBlock(out_channels, out_channels)
        self.resnet_attention = ResnetAttentionBlock(out_channels, num_heads)

        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.up_sample = nn.ConvTranspose2d(in_channels//2, in_channels//2, 4, 2, 1) if up_sample else nn.Identity()

    def forward(self, x: Tensor, out_down: Tensor, t_emb: Tensor) -> Tensor:
        x = self.up_sample(x)
        x = torch.cat([x, out_down], dim=1)

        out = x
        resnet_input = out
        out = self.resnet_conv_1(out)
        out += rearrange(self.t_emb_layers(t_emb), 'b c -> b c 1 1')
        out = self.resnet_conv_2(out)
        out += self.residual_input_conv(resnet_input)

        out_attn = self.resnet_attention(out)
        out += out_attn

        return out
    
class UNet(nn.Module):
    def __init__(
            self,
            im_channels: int,
            down_channels: list = [32, 64, 128, 256],
            mid_channels: list = [256, 256, 128],
            t_emb_dim: int = 128,
            down_sample: list = [True, True, False],
            num_heads: int = 4
        ) -> None:
        super().__init__()
        self.im_channels = im_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.up_sample = list(reversed(down_sample))

        self.t_proj = TimeProjectionBlock(t_emb_dim)
        self.conv_in = nn.Conv2d(im_channels, down_channels[0], 3, 1, 1)

        self.downs = nn.ModuleList([
            UNetDownBlock(
                self.down_channels[i],
                self.down_channels[i+1],
                self.t_emb_dim,
                self.down_sample[i],
                num_heads
            )
            for i in range(len(self.down_channels)-1)
        ])

        self.mids = nn.ModuleList([
            UNetMidBlock(
                self.mid_channels[i],
                self.mid_channels[i+1],
                self.t_emb_dim,
                num_heads
            )
            for i in range(len(self.mid_channels)-1)
        ])

        self.ups = nn.ModuleList([
            UNetUpBlock(
                self.down_channels[i]*2,
                self.down_channels[i-1] if i!=0 else 16,
                self.t_emb_dim,
                self.down_sample[i],
                num_heads
            )
            for i in reversed(range(len(self.down_channels)-1))
        ])

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, 3, 1, 1)

    def forward(self, x: Tensor, t: Tensor)->Tensor:
        assert x.shape[1] == self.im_channels, f'Input must have {self.im_channels} channels'
        out = self.conv_in(x)
        t_emb = self.t_proj(SinusoidalPositionalEncodingBlock(self.t_emb_dim)(t))

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)
            
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)    
        out = self.conv_out(out)

        return out

if __name__ == "__main__":
    unet = UNet(im_channels=1)
    x = torch.randn(4, 1, 28, 28)
    # random time steps in one dimension
    t = torch.randint(0, 100, (4,))
    print(t.shape)
    print(unet(x, t).shape)