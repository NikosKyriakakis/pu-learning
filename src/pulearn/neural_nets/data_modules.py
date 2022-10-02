from torch.utils.data import Dataset, DataLoader
from pulearn.pubase import extract_sample
from utils.download import *
from textprep.vectorizer import *
from textprep.embed import EmbeddingsHandler
from typing import Optional

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
import torch


class DeceptiveOpinionsDataset(Dataset):
    def __init__(self, documents) -> None:
        self.documents = documents
        
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents.iloc[index, :]
        sequence = torch.tensor(document["sequences"])
        label = torch.tensor(document["label"])
        return sequence, label
        

class DeceptiveOpinionsDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=64, num_workers=int(os.cpu_count() / 2), pretrained_embed="fasttext-crawl") -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrained_embed = pretrained_embed
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

    def _split_documents(self, sep_value):
        # Isolate documents 
        subset_docs = self.documents.loc[(self.documents["label"] == sep_value)]
        # Reset the index to avoid indices being not found later
        subset_docs.reset_index(drop=True, inplace=True)
        # Get indices of 20% of deceptive documents
        indices = extract_sample(subset_docs["label"], ratio=0.2, value=sep_value)
        # Get the required set
        subset_docs = subset_docs.loc[indices]
        
        return subset_docs

    def prepare_data(self) -> None:
        download_from_gdrive("deceptive-opinion.csv")

        self.documents = pd.read_csv("../data/deceptive-opinion.csv")
        self.documents["deceptive"] = self.documents["deceptive"].apply(lambda x: 0 if x == "truthful" else 1)
        self.documents["label"] = self.documents["deceptive"].copy()
        self.documents = self.documents.drop(columns=["hotel", "source", "polarity", "deceptive"])

        self.documents["sequences"], self.documents["text"], vectorizer = Vectorizer.from_dataframe (
            self.documents, 
            "text", 
            "label", 
            prep_funcs={
                to_lower: {},
                to_remove_symbols: {}
            }
        )

        emb_handler = EmbeddingsHandler(vectorizer.text_vocab, pretrained=self.pretrained_embed)
        self.embeddings = emb_handler.load_embeddings()
        
    def setup(self, stage: Optional[str] = None) -> None:
        # Get deceptive documents & non deceptive documents
        deceptive = self._split_documents(sep_value=1)
        non_deceptive = self._split_documents(sep_value=0)
        
        # Concatenate them to form a unified test set
        test_set = pd.concat([deceptive, non_deceptive], ignore_index=True)
        
        # Extract all documents which do not belong to the test set
        train_set = self.documents.loc[~(self.documents["text"].isin(test_set["text"]))]
        # Reset index to avoid indices not found error
        train_set = train_set.reset_index(drop=True)
        # Extract about 1040 elements to set as unlabeled
        indices = extract_sample(train_set["label"], ratio=0.37, value=1)
        # Mark labeled & unlabeled examples
        train_set["label"] = 0
        train_set.loc[indices, "label"] = 1

        # Split the test set in half to create a validation set
        pos_indices = extract_sample(test_set["label"], ratio=0.5, value=1)
        un_indices = extract_sample(test_set["label"], ratio=0.5, value=0)
        indices = np.concatenate((pos_indices, un_indices))
        
        val_set = test_set.loc[indices]
        val_set = val_set.reset_index(drop=True)

        test_set = test_set.drop(indices)
        test_set = test_set.reset_index(drop=True)

        self.train_set = DeceptiveOpinionsDataset(train_set[["sequences", "label"]])
        self.val_set = DeceptiveOpinionsDataset(val_set[["sequences", "label"]])
        self.test_set = DeceptiveOpinionsDataset(test_set[["sequences", "label"]])

    def train_dataloader(self):
        return DataLoader(self.train_set, **self._params)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self._params)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self._params)