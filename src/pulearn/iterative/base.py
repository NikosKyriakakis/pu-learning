from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import *
from sklearn.svm import SVC

from pulearn.pubase import *
from console import *

import pandas as pd
import numpy as np


def estimator_factory(option, params):
    models = {
        "Rocchio": (lambda: NearestCentroid(**params)),
        "SVC": (lambda: SVC(**params)),
        "GaussianNB": (lambda: GaussianNB(**params))
    }

    if option in models.keys():
        estimator = models[option]
        return estimator
    else:
        raise ValueError(error("Classifier options = {} --> Provided was: {}".format(models.keys(), option)))
    

def isolate_reliable_sample(X, y, predictions):
    """ Extract a reliable negative sample

    Args:
        X (pd.DataFrame): the data's features
        y (pd.Series): the target values
        predictions (nd.array): the results from which to retrieve the negative samples from

    Returns:
        tuple: the initial params minus the negative samples along with the latter
    """

    # Extract negative prediction indices
    indices = np.where(predictions == 0)[0]
    if indices.size == 0:
        return None
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Select the samples based on the indices
        Qx = X.iloc[indices, :]
        Qy = y[indices]

        X_tmp = X.drop(indices)
        y_tmp = y.drop(indices)

        return X_tmp, y_tmp, Qx, Qy


class IterativeClassifier(PUClassifier):
    def __init__(self, est_name1, est_name2, params1={}, params2={}) -> None:
        super().__init__()
        self._estimator = None
        self._best_model = None
        self.est_name1 = est_name1
        self.est_name2 = est_name2
        self.params1 = params1
        self.params2 = params2
    
    @abstractmethod
    def _update_negatives(self, predictions):
        """ Method to update the reliabe negative sets """

    @abstractmethod
    def _predict_partial(self):
        """ Make a prediction either on unlabeled or reliably negative data """

    def fit(self, X, y):
        # Separate positive & unlabeled sets
        self.Px, \
        self.Py, \
        self.Ux, \
        self.Uy = separate_sets(X, y)

        # Create initial model & fit on the whole data (P+U)
        self._estimator = estimator_factory(self.est_name1, self.params1)()
        self._estimator.fit(X, y)

        # Use the estimator on the unlabeled set
        predictions = self._estimator.predict(self.Ux)

        self.Ux, self.Uy, Qx, Qy = isolate_reliable_sample(self.Ux, self.Uy, predictions)

        self._prev_Q = self.RNx = Qx.copy()
        self.RNy = Qy.copy()

        count = 0
        while True:
            print(hourglass("Iteration: {}".format(count)))

            # Initialize the model
            self._estimator = estimator_factory(self.est_name2, self.params2)()

            # Use positives and reliable negatives as input to the actual classifier
            Dx = pd.concat([self.Px, self.RNx], ignore_index=True)
            Dy = pd.concat([self.Py, self.RNy], ignore_index=True)
            self._estimator.fit(Dx, Dy)

            # Save the first iteration
            # in case we need to rollback 
            # to this version
            if count == 0:
                self._best_model = self._estimator
            count += 1
                
            predictions = self._predict_partial()
            print(np.array(np.unique(predictions, return_counts=True)).T)
            status = self._update_negatives(predictions)
            if status == 1:
                break

        # Test the final classifier on the 
        # initial positive set
        predictions = self._estimator.predict(self.Px)
        positive = len([x for x in predictions if x == 1])
        negative = len([x for x in predictions if x == 0])

        try:
            ratio = negative / positive
            # print("Negative/Positive ratio: {}".format(ratio))
        except ZeroDivisionError:
            # Dummy value
            ratio = 0.1

        # If the misclassified points are above 5% 
        # we will rollback to the cached model
        # Else we keep the final classifier
        if ratio < 0.05:
            self._best_model = self._estimator   

    def predict(self, X):
        # Make a prediction on the whole unlabeled set
        # using the best classifier so far
        predictions = self._best_model.predict(X)
        # Check out how many got classified with 0 or 1
        results = np.unique(predictions, return_counts=True)
        results = np.array(results).T
        print(results)
        
        return predictions

