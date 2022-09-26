from abc import ABC, abstractmethod

import numpy as np


def extract_sample(target, ratio):
    """ Select positive examples completely at random 

    Args:
        target (array like): the target values
        ratio (float, optional): the percentage of positive examples to keep. Defaults to 0.25.

    Returns:
        array like: a subset of the positive examples
    """

    # Keep only positive indices and shuffle them
    positive_indices = np.where(target.values == 1)[0]
    np.random.shuffle(positive_indices)

    # Cut off a specified percentage of the positive examples
    threshold = int(np.ceil(ratio * len(positive_indices)))

    sample = positive_indices[:threshold]

    return sample


class PUClassifier(ABC):
    def __init__(self) -> None:
        """ Base PU learning class """

    @abstractmethod
    def fit(self, X, y, ratio=0.2):
        """ Override to specify training behaviour """

    @abstractmethod
    def predict(self, X):
        """ Override to specify testing behaviour """
