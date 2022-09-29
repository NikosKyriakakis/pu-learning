from abc import ABC, abstractmethod
from xmlrpc.client import boolean

import numpy as np
import pandas as pd

def _isolate_subset(X, y, value):
    indices = np.where(y.values == value)[0]
    subset_x = X.iloc[indices, :]
    subset_y = y[indices]

    subset_x = subset_x.reset_index(drop=True)
    subset_y = subset_y.reset_index(drop=True)

    return subset_x, subset_y
    
    
def separate_sets(X, y):
    """ Separate a PU dataset into a group of only positives and one of unlabeled points

    Args:
        X (pd.DataFrame): the data points to separate
        y (pd.Series): the pu-labels

    Returns:
        tuple: the separated positive and unlabeled sets 
    """

    positive_x, positive_y = _isolate_subset(X, y, 1)
    unlabeled_x, unlabeled_y = _isolate_subset(X, y, 0)

    return positive_x, positive_y, unlabeled_x, unlabeled_y


def extract_sample(target, ratio, value):
    """ Select positive examples completely at random 

    Args:
        target (array like): the target values
        ratio (float, optional): the percentage of positive examples to keep. Defaults to 0.25.

    Returns:
        array like: a subset of the positive examples
    """

    # Keep only positive indices and shuffle them
    positive_indices = np.where(target.values == value)[0]
    np.random.shuffle(positive_indices)

    # Cut off a specified percentage of the positive examples
    threshold = int(np.ceil(ratio * len(positive_indices)))

    sample = positive_indices[:threshold]

    return sample


class PUClassifier(ABC):
    def __init__(self) -> None:
        """ Base PU learning class """
        self.is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value):
        if type(value) != boolean:
            raise ValueError("[o_O] Only 'True' or 'False' allowed in boolean variables")
        else:
            self._is_fitted = value

    @abstractmethod
    def fit(self, X, y):
        """ Override to specify training behaviour """

    @abstractmethod
    def predict(self, X):
        """ Override to specify testing behaviour """
