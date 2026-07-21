from dataclasses import dataclass

@dataclass
class ModelParams:
    """Model parameters for the transformer model."""
    d_model: int = 512
    max_len: int = 5000

    in_vocab_size: int = 10000
    out_vocab_size: int = 10000
    embedding_dim: int = 512
    padding_idx: int = 0
    max_norm: float = None
    norm_type: int = 2.0 # 2-norm

    encoder_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8

