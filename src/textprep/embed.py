from tqdm import tqdm
from utils.download import download_embeddings

import numpy as np


class EmbeddingsHandler:
    def __init__(self, vocab, pretrained=None, dim=300) -> None:
        self.vocab = vocab
        self.embeddings = None
        self.dim = dim

        self._map = {
            "fasttext-wiki": "../embeddings/fasttext-wiki/wiki-news-300d-1M.vec",
            "fasttext-crawl": "../embeddings/fasttext-crawl/crawl-300d-2M.vec",
            "glove50": "../embeddings/glove/glove.6B.50d.txt",
            "glove100": "../embeddings/glove/glove.6B.100d.txt",
            "glove200": "../embeddings/glove/glove.6B.200d.txt",
            "glove300": "../embeddings/glove/glove.6B.300d.txt"
        }

        if pretrained is not None and pretrained in self._map:
            if "glove" in pretrained:
                option = "glove"
            else:
                option = pretrained

            download_embeddings(option)

            self.filename = self._map[pretrained]
        else:
            self.filename = None


    def load_embeddings(self, mode="r", encoding="utf-8", newline="\n", errors="ignore", dim=300):
        pad_token = self.vocab.pad_token
        embeddings = None
        
        with open(self.filename, mode, encoding=encoding, newline=newline, errors=errors) as file_ref:
            if "fasttext" in self.filename:
                file_ref.readline()
            
            embeddings = np.random.uniform(-0.25, 0.25, (len(self.vocab.word2index), dim))
            embeddings[self.vocab.lookup_token(pad_token)] = np.zeros((dim,))

            for line in tqdm(file_ref):
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if self.vocab.lookup_token(word) != self.vocab.unk_index:
                    embeddings[self.vocab.lookup_token(word)] = np.array(tokens[1:], dtype=np.float32)

        self.embeddings = embeddings