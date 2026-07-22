from datasets import load_dataset, load_from_disk
from configs import DatasetParams
# class Dataset:
#     """Dataset class for loading and processing datasets."""

#     def __init__(self, dataset_name: str):
#         self.dataset_name = dataset_name
#         self.dataset = load_dataset(dataset_name)

#     def get_dataset(self):
#         """Returns the loaded dataset."""
#         return self.dataset

class TagalogEnglishDataloader():
    """Dataloader class for loading and processing the Tagalog-English dataset."""

    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name, split=split)    


if __name__ == "__main__":
    dp = DatasetParams(dataset_name="rhyliieee/tagalog-filipino-english-translation")
    ds = load_dataset(dp.dataset_name)
    ds.save_to_disk(f"data/{dp.dataset_name}")

    print(ds['train'][0])

    
