import numpy as np
import pytest
import torch
from mockito import when, mock, unstub

from src.configuration import DownloadManager
from src.textprep.embed import EmbeddingLoader
from src.textprep.vocab import Vocabulary


@pytest.fixture
def embeddings_loader():
    settings = {
        "embedding_options": {
            "glove": {
                "command": "wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip",
                "filename": "glove.6B.zip"
            },
            "fasttext-wiki": {
                "command": "wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki"
                           "-news-300d-1M.vec.zip",
                "filename": "wiki-news-300d-1M.vec.zip"
            },
            "fasttext-crawl": {
                "command": "wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl"
                           "-300d-2M.vec.zip",
                "filename": "crawl-300d-2M.vec.zip"
            }
        },
        "dataset_options": {
            "deceptive-opinion.csv": "https://drive.google.com/uc?id=1QaV8r3l4EohQACCiwORr6Hqb9HQVhXV7",
            "PU_20News.csv": "https://drive.google.com/uc?id=1mKv0W_s8nrYyIhMDNU1I58MiLWNY32K5",
            "imdb.txt": "https://drive.google.com/uc?id=1z7-47JcSpt1WikNUqz4aWKOZqm9Lg188"
        },
        "data_dir": "../data",
        "embedding_dir": "../embeddings",
        "pretrained_embedding": {
            "option": "glove.6B.100d",
            "dim": 100
        }
    }
    vocab = Vocabulary(add_unk=True, add_pad=True)
    mgr = DownloadManager(settings)
    return EmbeddingLoader(vocab, mgr)


def test_init_embeddings_invalid_dim_raises_user_warning(embeddings_loader):
    with pytest.raises(UserWarning):
        embeddings_loader.init_embeddings(-1, "glove.6B.100d")


def test_init_embeddings_return_random_embeddings(embeddings_loader):
    expected = mock(torch.Tensor([np.random.uniform(1)]))
    when(embeddings_loader).init_embeddings(...).thenReturn(expected)
    actual = embeddings_loader.init_embeddings(100, "random")
    assert actual == expected

    unstub()


def test_init_embeddings_return_pretrained_embeddings(embeddings_loader):
    expected = mock(torch.Tensor([np.random.uniform(1)]))
    when(embeddings_loader).init_embeddings(...).thenReturn(expected)
    actual = embeddings_loader.init_embeddings(100, "glove.6B.100d")
    assert actual == expected

    unstub()


def test_init_embeddings_invalid_embeddings_option_raises_valuerror(embeddings_loader):
    with pytest.raises(ValueError):
        embeddings_loader.init_embeddings(100, "fake-embeddings")


def test_init_embeddings_embeddings_option_not_in_map_raises_valuerror(embeddings_loader):
    with pytest.raises(ValueError):
        embeddings_loader.init_embeddings(100, "glove-fake")


def test_init_embeddings_dim_inconsistent_with_file_raises_user_warning(embeddings_loader):
    with pytest.raises(UserWarning):
        when(embeddings_loader).init_embeddings(...).thenRaise(UserWarning)
        embeddings_loader.init_embeddings(100, "glove.6B.50d")
        unstub()
