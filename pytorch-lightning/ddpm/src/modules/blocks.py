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
    def __init__(self, t_emb_dim: int) -> None:
        super().__init__()
        self.t_emb_dim = t_emb_dim

    def forward(self, t) -> Tensor:
        factor = 1e4 ** ((
            torch.arange(
                start = 0,
                end = self.t_emb_dim // 2,
                device = t.device,
                ) / (self.t_emb_dim // 2)
            )
        )
        t_emb = repeat(rearrange(t, 't -> t 1'), 't 1 -> t d', d = self.t_emb_dim // 2) / factor
        return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)