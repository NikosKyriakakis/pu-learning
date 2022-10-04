from torch.utils.data import Dataset, DataLoader
from pulearn.pubase import extract_sample
from sklearn.model_selection import train_test_split
from typing import Optional

import os
import pytorch_lightning as pl
import torch


class DeceptiveOpinionsDataset(Dataset):
    def __init__(self, documents) -> None:
        self.documents = documents
        
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents.iloc[index, :]
        sequence = torch.tensor(document["sequences"])
        label = torch.tensor(document["pu-label"])
        return sequence, label
        

class DeceptiveOpinionsDataModule(pl.LightningDataModule):

    def __init__(self, documents, batch_size=128, num_workers=int(os.cpu_count() / 2)) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.documents = documents
        self._params = {"batch_size":batch_size, "num_workers":num_workers}
        self.prepare_data_per_node = True
        self._log_hyperparams = True

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value > 0 and type(value) == int:
            self._batch_size = value
        else:
            raise ValueError("[@_@] Provided batch size was either negative or a non-integer")
        
    def setup(self, stage: Optional[str] = None) -> None:
        indices = extract_sample(self.documents["label"], ratio=0.3, value=1)
        self.documents["pu-label"] = -1
        self.documents.loc[indices, "pu-label"] = 1
        self.documents = self.documents.reset_index(drop=True)

        pu_set = self.documents[["sequences", "pu-label"]]

        train_set, test_set = train_test_split(pu_set, test_size=0.2)
        val_set, test_set = train_test_split(test_set, test_size=0.5)

        self.train_set = DeceptiveOpinionsDataset(train_set)
        self.val_set = DeceptiveOpinionsDataset(val_set)
        self.test_set = DeceptiveOpinionsDataset(test_set)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self._params)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self._params)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self._params)