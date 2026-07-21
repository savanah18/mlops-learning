from dataclasses import dataclass

@dataclass
class DatasetParams:
    """Dataset parameters for the transformer model."""
    dataset_name: str = "rhyliieee/tagalog-filipino-english-translation"

@dataclass
class TokenizerParams:
    """Tokenizer parameters for the transformer model."""
    tokenizer_name: str = "basic_english"
    tokenizer_args: dict = None  # Additional arguments for the tokenizer, if needed
    tokenizer_kwargs: dict = None  # Additional keyword arguments for the tokenizer, if needed