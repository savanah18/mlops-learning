from datasets import load_dataset

class Dataset:
    """Dataset class for loading and processing datasets."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)

    def get_dataset(self):
        """Returns the loaded dataset."""
        return self.dataset

if __name__ == "__main__":
    ds = load_dataset("rhyliieee/tagalog-filipino-english-translation")