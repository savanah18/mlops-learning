import torch
from torch import nn

from configs import ModelParams
from blocks import PositionalEncoding, EncoderBlock, DecoderBlock



class Encoder():
    def __init__(self):
        pass

class Decoder():
    def __init__(self):
        pass

# Traditional Seq2Seq Translator 
class Translator(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.model_params = model_params

        # Define the embedding layer
        self.in_embedding = nn.Embedding(
            num_embeddings=model_params.in_vocab_size,
            embedding_dim=model_params.embedding_dim,
            padding_idx=model_params.padding_idx,
            max_norm=model_params.max_norm,
            norm_type=model_params.norm_type
            ) # Bx* -> B x * x embedding_dim
        
        self.out_embedding = nn.Embedding(
            num_embeddings=model_params.out_vocab_size,
            embedding_dim=model_params.embedding_dim,
            padding_idx=model_params.padding_idx,
            max_norm=model_params.max_norm,
            norm_type=model_params.norm_type
            ) # Bx* -> B x * x embedding_dim

        # define the positional encoding layer
        self.pe = PositionalEncoding(d_model=model_params.d_model, max_len=model_params.max_len).pe

        # define the encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(model_params=model_params) for _ in range(model_params.encoder_layers)]
        )

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(model_params=model_params) for _ in range(model_params.decoder_layers)]
        )

        # final linear layer
        self.linear = nn.Linear(
            in_features=model_params.d_model,
            out_features=model_params.out_vocab_size
        )
        self.logits = nn.Softmax(dim=-1)  #softmax in the last dimension to get the probabilities of each token in the vocabulary


    def forward(self, src, tgt):
        src_emb = self.in_embedding(src) + self.pe[:src.size(1), :]  # Add positional encoding to the source embeddings
        tgt_emb = self.out_embedding(tgt) + self.pe[:tgt.size(1), :]  # Add positional encoding to the target embeddings

        assert src_emb.shape[-1] == self.model_params.d_model, f"Source embedding dimension {src_emb.shape[-1]} does not match model dimension {self.model_params.d_model}"
        assert tgt_emb.shape[-1] == self.model_params.d_model, f"Target embedding dimension {tgt_emb.shape[-1]} does not match model dimension {self.model_params.d_model}"

        x,y = src_emb, tgt_emb
        enc_x = None

        for blk_idx in range(self.model_params.encoder_layers):
            enc_x = self.encoder_blocks[blk_idx](x, x, x)  # Self-attention for 
            y = self.decoder_blocks[blk_idx](y, y, y, enc_x)  # Self-attention for decoder and cross-attention with encoder output
            x = enc_x  # Update x for the next encoder block

        return self.logits(self.linear(y)) 


if __name__ == "__main__":
    model_params = ModelParams()
    source = torch.randint(1,model_params.in_vocab_size, (2, 10)) #Bx*
    target = torch.LongTensor([[1], [1]]) #Bx*
    translator = Translator(model_params=ModelParams())

    max_token = 100 #instead of EOS
    # with torch.no_grad():
    #     for _ in range(max_token):
    #         translated = translator(source, target)
    #         assert translated.shape == (2, 1, model_params.out_vocab_size), f"Output shape {translated.shape} does not match expected shape {(2, 1, model_params.out_vocab_size)}"

    #         # no need to beam search, or greedy search, just take the argmax for now
    #         translate_tokens = torch.argmax(translated, dim=-1)
    #         print(f"Translated tokens: {translate_tokens}")
    #         torch.concat((target, translate_tokens), dim=-1) # Bx* -> Bx*+1
    #         print(target.shape)

    with torch.no_grad():
        for step in range(max_token):
            # 1. Pass the current target sequence into the model
            translated = translator(source, target) 
            next_token_logits = translated[:, -1:, :] 
            next_token = torch.argmax(next_token_logits, dim=-1) # Shape: (2, 1)
            target = torch.cat([target, next_token], dim=1)
    print(target.shape)
