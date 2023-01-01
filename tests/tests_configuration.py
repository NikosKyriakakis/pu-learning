import pytest
from mockito import when, mock, unstub

import src.configuration as cfg


@pytest.fixture
def download_manager():
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

    return cfg.DownloadManager(settings)


def test_load_settings_file_found():
    expected_settings = mock({"fake-key": "fake-value"})
    when(cfg).load_settings(...).thenReturn(expected_settings)

    actual_settings = cfg.load_settings("fake-settings.json")
    assert actual_settings == expected_settings

    unstub()


def test_load_settings_file_not_found():
    with pytest.raises(FileNotFoundError):
        cfg.load_settings("fake-settings.json")


def test_extract_file_not_found():
    with pytest.raises(IOError):
        cfg.extract_file("fake.zip", "fake-output-dir")


def test_download_from_gdrive_invalid_dataset(download_manager):
    with pytest.raises(UserWarning):
        download_manager.download_from_gdrive("fake-data.csv")


def test_download_embeddings_invalid_option(download_manager):
    with pytest.raises(UserWarning):
        download_manager.download_embeddings("fake-embedding")
