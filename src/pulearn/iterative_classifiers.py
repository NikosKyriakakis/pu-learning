from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from copy import deepcopy
from pulearn.pubase import *

import pandas as pd
import numpy as np


def estimator_factory(option, params):
    models = {
        "Rocchio": (lambda: NearestCentroid(**params)),
        "SVC": (lambda: SVC(**params))
    }

    if option in models.keys():
        estimator = models[option]
    else:
        estimator = None
    return estimator


class TwostepClassifier(PUClassifier):
    def __init__(self, est_name1, est_name2, params1={}, params2={}) -> None:
        super().__init__()
        self._estimator = None
        self.est_name1 = est_name1
        self.est_name2 = est_name2
        self.params1 = params1
        self.params2 = params2


    def isolate_reliable_sample(self, Ux, Uy, value):
        # Use the estimator on the unlabeled set
        predictions = self._estimator.predict(Ux)
        # Extract negative prediction indices
        indices = np.where(predictions == value)[0]
        if indices.size == 0:
            return None
        else:
            # Select the samples based on the indices
            Qx = Ux.iloc[indices, :]
            Qy = Uy[indices]
            # Remove extracted negatives from unlabeled data
            Ux = Ux.drop(indices)
            Uy = Uy.drop(indices)
            Ux = Ux.reset_index(drop=True)
            Uy = Uy.reset_index(drop=True)

            return Ux, Uy, Qx, Qy


    def fit(self, X, y):
        # Separate positive & unlabeled sets
        Px, \
        Py, \
        Ux, \
        Uy = separate_sets(X, y)

        # Create a copy to retain a separate 
        # test set when the model has been fitted
        U_test = Ux.copy()

        # Create initial model & fit on the whole data (P+U)
        self._estimator = estimator_factory(self.est_name1, self.params1)()
        self._estimator.fit(X, y)

        Ux, Uy, Qx, Qy = self.isolate_reliable_sample(Ux, Uy, 0)

        RNx = Qx.copy()
        RNy = Qy.copy()

        # We want to retain a copy of 
        # the original positives
        Dx = Px.copy()

        count = 0
        while True:
            # Initialize the model
            self._estimator = estimator_factory(self.est_name2, self.params2)()

            # Use positives and reliable negatives as input to the actual classifier
            Dx = pd.concat([Dx, RNx], ignore_index=True)
            Py = pd.concat([Py, RNy], ignore_index=True)
            self._estimator.fit(Dx, Py)

            # Save the first iteration
            # in case we need to rollback 
            # to this version
            if count == 0:
                self._best_model = deepcopy(self._estimator)

            print("Iteration: {}".format(count))
            print("Unlabeled points remaining: {}\n".format(Ux.shape[0]))
            count += 1
                
            values = self.isolate_reliable_sample(Ux, Uy, 0)
            if values is None:
                break
            
            Ux, Uy, Qx, Qy = values

            if Ux.empty:
                break
            
            # Merge previous reliable negatives with the 
            # newly acquired ones
            RNx = pd.concat([RNx, Qx], ignore_index=True)
            RNy = pd.concat([RNy, Qy], ignore_index=True)

        # Test the final classifier on the 
        # initial positive set
        predictions = self._estimator.predict(Px)
        positive = len([x for x in predictions if x == 1])
        negative = len([x for x in predictions if x == 0])

        try:
            ratio = negative / positive
            print("Negative/Positive ratio: {}".format(ratio))
        except ZeroDivisionError:
            # Dummy value
            ratio = 0.1

        # If the misclassified points are above 5% 
        # we will rollback to the cached model
        # Else we keep the final classifier
        if ratio < 0.05:
            self._best_model = self._estimator   

        return U_test

    def predict(self, X):
        # Make a prediction on the whole unlabeled set
        # using the best classifier so far
        predictions = self._best_model.predict(X)
        # Check out how many got classified with 0 or 1
        results = np.unique(predictions, return_counts=True)
        results = np.array(results).T
        print(results)
        
        return predictions