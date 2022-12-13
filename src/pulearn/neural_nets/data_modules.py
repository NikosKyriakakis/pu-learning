from torch.utils.data import Dataset, DataLoader
from pulearn.pubase import extract_sample
from sklearn.model_selection import train_test_split
from typing import Optional
from textprep.vectorizer import *
from configuration import *
from console import *

import os
import pandas as pd
import pytorch_lightning as pl
import torch


class TextDataset(Dataset):
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
        

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        csv_file, 
        input_field, 
        target_field, 
        download_mgr,
        negative_value,
        keep_positive=0.3,
        vectorizer_factory=SequenceVectorizer,
        prep_funcs={
            to_lower: {}, 
            to_remove_symbols: {}
        },
        dataloader_params={
            "batch_size": 32, 
            "num_workers": int(os.cpu_count() / 2)
        },
        datasplit_params={
            "test_size": 0.4,
            "val_size": 0.5
        }
    ) -> None:

        self.csv_file = csv_file
        self.input_field = input_field
        self.target_field = target_field

        self.negative_value = negative_value
        self.keep_positive = keep_positive

        self.download_mgr = download_mgr
        self.vectorizer_factory = vectorizer_factory
        self.prep_funcs = prep_funcs
        self.datasplit_params = datasplit_params

        self.prepared_flag = False
        self._dataloader_params = dataloader_params
        self.prepare_data_per_node = True
        self._log_hyperparams = True
        self.prepare_data()

    def prepare_data(self) -> None:
        if self.prepared_flag:
            return

        self.download_mgr.download_from_gdrive(self.csv_file)

        data_dir = self.download_mgr.settings["data_dir"]
        data_path = os.path.join(data_dir, self.csv_file)

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

        if type(self.negative_value) == int:
            self.documents["label"] = self.documents["label"].apply(lambda x: 0 if int(x) == self.negative_value else 1)
        elif type(self.negative_value) == str:
            self.documents["label"] = self.documents["label"].apply(lambda x: 0 if x == self.negative_value else 1)
        else:
            raise UserWarning(error("Invalid label type provided"))
        
        indices = extract_sample(self.documents["label"], ratio=self.keep_positive, value=1)
        self.documents["pu-label"] = 0
        self.documents.loc[indices, "pu-label"] = 1
        self.documents = self.documents.reset_index(drop=True)

        self.vectorizer = self.vectorizer_factory.from_dataframe (
            data=self.documents, 
            prep_funcs=self.prep_funcs
        )

        self.prepared_flag = True
            
    def setup(self, stage: Optional[str] = None) -> None:
        pu_set = self.documents[["text", "pu-label"]]

        test_size = self.datasplit_params["test_size"]
        val_size = self.datasplit_params["val_size"]

        train_set, test_set = train_test_split(pu_set, test_size=test_size)
        val_set, test_set = train_test_split(test_set, test_size=val_size)

        self.train_set = TextDataset(train_set, self.vectorizer)
        self.val_set = TextDataset(val_set, self.vectorizer)
        self.test_set = TextDataset(test_set, self.vectorizer)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self._dataloader_params)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self._dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self._dataloader_params)