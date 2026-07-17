import torch
from torch import nn

from .configs import ModelParams
from .blocks import PositionalEncoding, EncoderBlock, DecoderBlock



class Encoder():
    def __init__(self):
        pass

class Decoder():
    def __init__(self):
        pass

class Transformer(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.model_params = model_params

        # Define the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=model_params.vocab_size,
            embedding_dim=model_params.embedding_dim,
            padding_idx=model_params.padding_idx,
            max_norm=model_params.max_norm,
            norm_type=model_params.norm_type
            ) # Bx* -> B x * x embedding_dim
        
        # define the positional encoding layer
        self.pe = PositionalEncoding(d_model=model_params.d_model, max_len=model_params.max_len).pe




if __name__ == "__main__":
    torch