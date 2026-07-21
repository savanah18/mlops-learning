import torch
from torch import nn

class PositionalEncoding():
    # cosine/sine implementation
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model)

        # 1 x L --> 0,5000
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # e^{-2i*log10000/d_model} = 10000^{-2i/d_model} = 1/1000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

class EncoderBlock(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.mha = nn.MultiheadAttention(
            embed_dim=model_params.d_model, 
            num_heads=model_params.num_heads,
            batch_first=True)
        self.norm1 = nn.LayerNorm(model_params.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(model_params.d_model, model_params.d_model * 4),
            nn.ReLU(),
            nn.Linear(model_params.d_model * 4, model_params.d_model)
        )
        self.norm2 = nn.LayerNorm(model_params.d_model)

    def forward(self, q, k, v):
        # mha returns both output and attention weights, we only need the output
        x = self.norm1(self.mha(q, k, v)[0]) + q # modern arch norm first before residual connection
        x = self.norm2(self.ffn(x) ) + x 
        return x


class DecoderBlock(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.mha1 = nn.MultiheadAttention(
            embed_dim=model_params.d_model, 
            num_heads=model_params.num_heads,
            batch_first=True)
        self.norm1 = nn.LayerNorm(model_params.d_model)

        self.mha2 = nn.MultiheadAttention(
            embed_dim=model_params.d_model, 
            num_heads=model_params.num_heads,
            batch_first=True)
        self.norm2 = nn.LayerNorm(model_params.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(model_params.d_model, model_params.d_model * 4),
            nn.ReLU(),
            nn.Linear(model_params.d_model * 4, model_params.d_model)
        )
        self.norm3 = nn.LayerNorm(model_params.d_model)

    def forward(self, q, k, v, enc_x):
        x = self.norm1(self.mha1(q, k, v)[0]) + q # modern arch norm first before residual connection
        x = self.norm2(self.mha2(x, enc_x, enc_x)[0]) + x 
        x = self.norm3(self.ffn(x) ) + x 
        return x

if __name__ == "__main__":
    from configs import ModelParams
    model_params = ModelParams()
    x=torch.rand(2, model_params.max_len, model_params.d_model) #BxLxd
    y=torch.rand(2, model_params.max_len, model_params.d_model) #BxLxd

    with torch.no_grad():
        encoder_block = EncoderBlock(model_params)
        out = encoder_block(x, x, x)
        assert out.shape == (2, model_params.max_len, model_params.d_model)

        decoder_block = DecoderBlock(model_params)
        out = decoder_block(y, y, y, out)
        assert out.shape == (2, model_params.max_len, model_params.d_model)