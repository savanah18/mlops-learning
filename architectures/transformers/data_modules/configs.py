from dataclasses import dataclass, field

@dataclass
class DatasetParams:
    """Dataset parameters for the transformer model."""
    dataset_name: str = "rhyliieee/tagalog-filipino-english-translation"

@dataclass
class TokenizerParams:
    """Tokenizer parameters for the transformer model."""
    tokenizer_name: str = "jcblaise/roberta-tagalog-base"
    tokenizer_args: dict = field(default_factory=dict) # Additional arguments for the tokenizer, if needed
    tokenizer_kwargs: dict = field(default_factory=dict) # Additional keyword arguments for the tokenizer, if needed