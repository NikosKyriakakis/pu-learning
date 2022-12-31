from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional

import os
import pandas as pd
import pytorch_lightning as pl
import torch

from src.console import error
from src.pulearn.utils import extract_sample
from src.textprep.vectorizer import SequenceVectorizer, to_lower, to_remove_symbols


class TextDataset(Dataset):
    def __init__(self, documents, vectorizer) -> None:
        self._documents = documents
        self._vectorizer = vectorizer

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index):
        document = self._documents.iloc[index, :]
        sequence = self._vectorizer.vectorize(document["text"])
        sequence = torch.tensor(sequence)
        label = torch.tensor(document["pu-label"])

        return sequence, label


class TextDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, input_field, target_field, download_mgr, negative_value, keep_positive=0.05,
                 vectorizer_factory=SequenceVectorizer, dev_run=False, prep_funcs=None, dataloader_params=None,
                 datasplit_params=None) -> None:

        super().__init__()

        self._documents = None
        self._test_set = None
        self._val_set = None
        self._train_set = None
        self._vectorizer = None

        if datasplit_params is None:
            datasplit_params = {
                "test_size": 0.4,
                "val_size": 0.5
            }

        if dataloader_params is None:
            dataloader_params = {
                "batch_size": 32,
                "num_workers": int(os.cpu_count() / 2)
            }

        if prep_funcs is None:
            prep_funcs = {
                to_lower: {},
                to_remove_symbols: {}
            }

        self._csv_file = csv_file
        self._input_field = input_field
        self._target_field = target_field
        self._dev_run = dev_run
        self._negative_value = negative_value
        self._keep_positive = keep_positive

        self._download_mgr = download_mgr
        self._vectorizer_factory = vectorizer_factory
        self._prep_funcs = prep_funcs
        self._datasplit_params = datasplit_params

        self._prepared_flag = False
        self._dataloader_params = dataloader_params
        self.__prepare_data_per_node = True
        self._log_hyperparams = True

        self._prepare_data()

    def _prepare_data(self) -> None:
        if self._prepared_flag:
            return

        self._download_mgr.download_from_gdrive(self._csv_file)

        data_dir = self._download_mgr.settings["data_dir"]
        data_path = os.path.join(data_dir, self._csv_file)

        self._documents = pd.read_csv(
            data_path,
            usecols=[
                self._input_field,
                self._target_field
            ],
            dtype={
                self._input_field: "string",
                self._target_field: "string"
            },
        )

        self._documents.rename(
            columns={
                self._input_field: "text",
                self._target_field: "label"
            },
            inplace=True
        )

        if type(self._negative_value) == int:
            self._documents["label"] = self._documents["label"]\
                .apply(lambda x: 0 if int(x) == self._negative_value else 1)
        elif type(self._negative_value) == str:
            self._documents["label"] = self._documents["label"]\
                .apply(lambda x: 0 if x == self._negative_value else 1)
        else:
            raise UserWarning(error("Invalid label type provided"))

        if self._dev_run:
            self._documents = self._documents.head(5000)

        indices = extract_sample(self._documents["label"], ratio=self._keep_positive, value=1)
        self._documents["pu-label"] = 0
        self._documents.loc[indices, "pu-label"] = 1
        self._documents = self._documents.reset_index(drop=True)

        self._vectorizer = self._vectorizer_factory.from_dataframe(
            data=self._documents,
            prep_funcs=self._prep_funcs
        )

        self._prepared_flag = True

    def setup(self, stage: Optional[str] = None) -> None:
        test_size = self._datasplit_params["test_size"]
        val_size = self._datasplit_params["val_size"]

        pu_set = self._documents[["text", "pu-label"]]

        train_set, test_set = train_test_split(pu_set, test_size=test_size)
        val_set, test_set = train_test_split(test_set, test_size=val_size)

        self._train_set = TextDataset(train_set, self._vectorizer)
        self._val_set = TextDataset(val_set, self._vectorizer)
        self._test_set = TextDataset(test_set, self._vectorizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_set, **self._dataloader_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_set, **self._dataloader_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_set, **self._dataloader_params)

    @property
    def vectorizer(self) -> SequenceVectorizer:
        return self._vectorizer

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents
