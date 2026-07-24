from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


from data_modules.configs import DatasetParams, TokenizerParams

def load_dataset(dataset_params: DatasetParams, online: bool = True, path: str = "./data_modules/data", split: str = "train"):
    """Load the training data from the specified dataset."""
    if online:
        print(f"Loading dataset from Hugging Face Hub: {dataset_params.dataset_name}")
        return  load_dataset(dataset_params.dataset_name)[split]
    else:
        print(dataset_params)
        print(f"Loading dataset from disk: {path}/{dataset_params.dataset_name}")
        load_from_disk(f"{path}/{dataset_params.dataset_name}")
        return  load_from_disk(f"{path}/{dataset_params.dataset_name}")[split] 

def load_tokenizer(tokenizer_params: TokenizerParams, online: bool = True, path: str = "./data_modules/tokenizers"):
    """Load the tokenizer from the specified tokenizer name."""
    if online:
        print(f"Loading tokenizer from Hugging Face Hub: {tokenizer_params.tokenizer_name}")
        return AutoTokenizer.from_pretrained(tokenizer_params.tokenizer_name, **tokenizer_params.tokenizer_args, **tokenizer_params.tokenizer_kwargs)
    else:
        print(f"Loading tokenizer from disk: {path}/{tokenizer_params.tokenizer_name}")
        return AutoTokenizer.from_pretrained(f"{path}/{tokenizer_params.tokenizer_name}", **tokenizer_params.tokenizer_args, **tokenizer_params.tokenizer_kwargs)