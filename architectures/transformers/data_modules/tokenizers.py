# Some notes
# Below are some different tokenizers that can be used. 
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import (
    BertTokenizer, #Implements the WordPiece algorithm for tokenizing text, commonly used with BERT models
    RegexTokenizer, #Allows custom tokenization using regex patterns, ideal for specific preprocessing needs:
    SentencepieceTokenizer, #This tokenizer uses a pre-trained SentencePiece model for subword tokenization. It is useful for tasks like machine translation or summarization
    GPT2BPETokenizer, #Implements Byte Pair Encoding (BPE) for tokenization, as used in GPT-2
)

from configs import TokenizerParams

class CustomTokenizer(nn.Module):
    # TODO
    def __init(self):
        pass

    def forward(self, text: str):
        # TODO
        pass

class Tokenizer:
    def __init__(self, tokenizer_params: TokenizerParams):
        match tokenizer_params.tokenizer_name:
            case "basic_english":
                self.tokenizer = get_tokenizer("basic_english")
            case "bert":
                self.tokenizer = BertTokenizer(*(tokenizer_params.tokenizer_kwargs or {}))
            case "sentencepiece":
                self.tokenizer = SentencepieceTokenizer(*(tokenizer_params.tokenizer_kwargs or {}))
            case "regex":
                self.tokenizer = RegexTokenizer(**(tokenizer_params.tokenizer_kwargs or {}))
            case "gpt2bpe":
                self.tokenizer = GPT2BPETokenizer(**(tokenizer_params.tokenizer_kwargs or {}))
            case "custom":
                self.tokenizer = CustomTokenizer()
            case _:
                raise ValueError(f"Unsupported tokenizer: {tokenizer_params.tokenizer_name}")

    def tokenize(self, text: str):
        return self.tokenizer(text)

if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer("You can now install TorchText using pip!")
    print(tokens) # Output: ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']