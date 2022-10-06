from torch.utils.data import Dataset, DataLoader
from pulearn.pubase import extract_sample
from sklearn.model_selection import train_test_split
from typing import Optional
from textprep.vectorizer import *
from utils.download import download_from_gdrive

import os
import pandas as pd
import pytorch_lightning as pl
import torch


class ReviewDataset(Dataset):
    def __init__(self, documents, vectorizer) -> None:
        self.documents = documents
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents.iloc[index, :]

        sequence = self.vectorizer.vectorize(document["text"])
        sequence = torch.tensor(sequence)

        label = torch.tensor(document["pu-label"])

        return sequence, label
        

class ReviewDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        csv_file, 
        input_field, 
        target_field, 
        negative_label="truthful",
        keep_positive=0.3,
        batch_size=128,
        num_workers=int(os.cpu_count() / 2),
        vectorizer_factory=SequenceVectorizer,
        prep_funcs={to_lower: {}, to_remove_symbols: {}}
    ) -> None:

        self.csv_file = csv_file
        self.input_field = input_field
        self.target_field = target_field

        self.negative_label = negative_label
        self.keep_positive = keep_positive

        self.vectorizer_factory = vectorizer_factory
        self.prep_funcs = prep_funcs

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepared_flag = False
        self._params = {"batch_size": batch_size, "num_workers": num_workers}
        self.prepare_data_per_node = True
        self._log_hyperparams = True

        self.prepare_data()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value > 0 and type(value) == int:
            self._batch_size = value
        else:
            raise ValueError("[@_@] Provided batch size was either negative or a non-integer")
        
    def prepare_data(self) -> None:
        
        if self.prepared_flag:
            return

        download_from_gdrive(self.csv_file)

        data_path = os.path.join("../data", self.csv_file)

        self.documents = pd.read_csv (
            data_path,
            usecols=[
                self.input_field, 
                self.target_field
            ], 
            dtype={
                self.input_field: "string",
                self.target_field: "string"
            }
        )

        self.documents.rename (
            columns={
                self.input_field: "text",
                self.target_field: "label"
            },
            inplace=True
        )

        if self.negative_label is not None:
            self.documents["label"] = self.documents["label"].apply(lambda x: -1 if x == self.negative_label else 1)

        indices = extract_sample(self.documents["label"], ratio=self.keep_positive, value=1)
        self.documents["pu-label"] = -1
        self.documents.loc[indices, "pu-label"] = 1
        self.documents = self.documents.reset_index(drop=True)

        self.vectorizer = self.vectorizer_factory.from_dataframe (
            data=self.documents, 
            prep_funcs=self.prep_funcs
        )

        self.prepared_flag = True
            

    def setup(self, stage: Optional[str] = None) -> None:
        pu_set = self.documents[["text", "pu-label"]]

        train_set, test_set = train_test_split(pu_set, test_size=0.2)
        val_set, test_set = train_test_split(test_set, test_size=0.5)

        self.train_set = ReviewDataset(train_set, self.vectorizer)
        self.val_set = ReviewDataset(val_set, self.vectorizer)
        self.test_set = ReviewDataset(test_set, self.vectorizer)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self._params)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self._params)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self._params)