import numpy as np
import pandas as pd


def _isolate_subset(x: pd.DataFrame, y: pd.Series, value: int) -> tuple:
    indices = np.where(y.values == value)[0]
    subset_x = x.iloc[indices, :]
    subset_x = subset_x.reset_index(drop=True)
    subset_y = y[indices]
    subset_y = subset_y.reset_index(drop=True)

    return subset_x, subset_y
    
    
def separate_sets(x: pd.DataFrame, y: pd.Series) -> tuple:
    positive_x, positive_y = _isolate_subset(x, y, 1)
    unlabeled_x, unlabeled_y = _isolate_subset(x, y, 0)

    return positive_x, positive_y, unlabeled_x, unlabeled_y


def extract_sample(target: pd.Series, ratio: float, value: int) -> np.array:
    positive_indices = np.where(target.values == value)[0]
    np.random.shuffle(positive_indices)
    threshold = int(np.ceil(ratio * len(positive_indices)))
    sample = positive_indices[:threshold]

    return sample
