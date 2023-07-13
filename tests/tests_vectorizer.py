import pandas as pd
import pytest

from src.textprep.vectorizer import SequenceVectorizer


@pytest.fixture
def normal_sequence_vectorizer():
    vectorizer = SequenceVectorizer.from_dataframe(
        pd.DataFrame(columns=["fake_text", "fake_labels"]),
        "fake_text",
        "fake_labels"
    )
    vectorizer.max_len = 10

    return vectorizer


def test_empty_sequence_vectorizer():
    with pytest.raises(ValueError):
        SequenceVectorizer.from_dataframe(pd.DataFrame(), "text", "labels")


def test_vectorize(normal_sequence_vectorizer):
    seq = normal_sequence_vectorizer.vectorize("hello world")
    assert seq == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
