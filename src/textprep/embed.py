from tqdm import tqdm
from os import listdir, walk
from os.path import join

import numpy as np
import torch

from configuration import DownloadManager
from console import error, hourglass, success
from textprep.vocab import Vocabulary


class EmbeddingLoader:
    def __init__(self, vocab: Vocabulary, mgr: DownloadManager) -> None:
        self.vocab = vocab
        self._map = {}
        self._mgr = mgr
        self.embedding_path = ""

    def _discover_embedding_subdirs(self) -> None:
        self._map = {}
        embedding_dir = self._mgr.settings["embedding_dir"]
        directories = next(walk(embedding_dir))[1]

        for directory in directories:
            relative_path = join(embedding_dir, directory)
            files = listdir(relative_path)
            for file in files:
                self._map[file[:-4]] = join(relative_path, file)

    def init_embeddings(
            self,
            dim: int,
            pretrained: str,
            mode: str = "r",
            encoding: str = "utf-8",
            newline: str = "\n",
            errors: str = "ignore"
    ) -> torch.Tensor:
        if dim <= 0:
            raise UserWarning(error("Zero or negative dimensions are not allowed."))

        embeddings = np.random.uniform(-0.25, 0.25, (len(self.vocab.word2index), dim))
        pad_token = self.vocab.pad_token
        embeddings[self.vocab.lookup_token(pad_token)] = np.zeros((dim,))

        if pretrained == "random":
            print(hourglass("Using random embeddings ..."))
            return torch.tensor(embeddings)

        if "glove" in pretrained:
            option = "glove"
        elif "wiki" in pretrained:
            option = "fasttext-wiki"
        elif "crawl" in pretrained:
            option = "fasttext-crawl"
        else:
            raise ValueError(error(
                "Invalid pretrained embedding name provided --> Check available pretrained embeddings in "
                "app-settings.json"))

        self._mgr.download_embeddings(option)
        self._discover_embedding_subdirs()
        if pretrained not in self._map:
            raise ValueError(error(
                "Pretrained embeddings option is not valid --> Check that filename is the same as in the embeddings "
                "folder."))
        self.embedding_path = self._map[pretrained]

        with open(self.embedding_path, mode, encoding=encoding, newline=newline, errors=errors) as file_ref:
            if "glove" not in self.embedding_path:
                file_ref.readline()

            print(hourglass("Loading {} pretrained embeddings ...".format(self.embedding_path.split("/")[-1])))

            try:
                for line in tqdm(file_ref):
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    if self.vocab.lookup_token(word) != self.vocab.unk_index:
                        embeddings[self.vocab.lookup_token(word)] = np.array(tokens[1:], dtype=np.float32)
            except ValueError:
                raise UserWarning(error(
                    "Check if selected embeddings file matches provided dimension (e.g., glove.6B.50d expects dim=50)"))

            print(success("Pretrained embeddings loaded successfully"))

        return torch.tensor(embeddings)
