import numpy as np
import pytest
import pandas as pd

from src.pulearn.utils import separate_sets, extract_sample


@pytest.fixture
def default_dataframe():
    data = np.random.uniform(-0.25, 0.25, (10, 2))
    data = pd.DataFrame(data)
    labels = pd.Series(5 * [0] + 5 * [1])
    return data, labels


def test_separate_sets(default_dataframe):
    data, labels = default_dataframe
    x_p, y_p, x_u, y_u = separate_sets(data, labels)
    assert len(x_p) == len(y_p) and len(x_u) == len(y_u)
    assert y_p.tolist() == [1, 1, 1, 1, 1]
    assert y_u.tolist() == [0, 0, 0, 0, 0]


def test_extract_sample(default_dataframe):
    _, labels = default_dataframe

    sample = extract_sample(labels, 0.1, 1)
    assert len(sample) == 1
    assert labels[sample[0]] == 1
