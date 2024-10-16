import torch
from torch import (
    nn,
    Tensor
)
from einops import (
    rearrange,
    repeat
)

class SinusoidalPositionalEncodingBlock(nn.Module):
    def __init__(self, timesteps, t_emb_dim: int) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.t_emb_dim = t_emb_dim

    def forward(self) -> Tensor:
        factor = 1e4 ** ((
            torch.arange(
                start = 0,
                end = self.t_emb_dim // 2,
                device = self.timesteps.device,
                ) / (self.t_emb_dim // 2)
            )
        )
        t_emb = repeat(rearrange(self.timesteps, 't -> t 1'), 't 1 -> t d', d = self.t_emb_dim // 2) / factor
        return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)

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
        self.attention_norm = nn.GroupNorm(8, in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim = in_channels,
            num_heads = 4,
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