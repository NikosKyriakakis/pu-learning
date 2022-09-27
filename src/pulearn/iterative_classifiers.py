from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from copy import deepcopy
from pulearn.pubase import *

import pandas as pd
import numpy as np


class RocSVM(PUClassifier):
    """ Initial implementation based on the paper 
            'Learning to Classify Texts Using Positive and Unlabeled Data'
    """
    def __init__(self) -> None:
        super().__init__()
        self._best_model = None


    def fit(self, X, y):
        # Separate the data into Positive & Unlabeled sets
        Px, Py, Ux, Uy = separate_sets(X, y)
        Ux_test = Ux.copy()

        # Create initial Rocchio classifier
        rocchio = NearestCentroid()
        rocchio.fit(X, y)

        # Create the U^L set (i.e. from this set we will extract reliably negative examples)
        UL = rocchio.predict(Ux)

        # Extract reliably negative samples
        indices = np.where(UL == 0)[0]
        Qx = Ux.iloc[indices, :]
        Qy = Uy[indices]

        # Create a copy of the Reliably Negatives
        RNx = Qx.copy()
        RNy = Qy.copy()

        # Remove extracted negatives from unlabeled data
        Ux = Ux.drop(indices)
        Uy = Uy.drop(indices)
        Ux = Ux.reset_index(drop=True)
        Uy = Uy.reset_index(drop=True)

        # We want to retain a copy of the original positives
        Dx = Px.copy()

        count = 0
        while True:
            # Initialize the model
            svc = SVC()

            # Use positives and reliable negatives as input to the SVM
            Dx = pd.concat([Dx, RNx], ignore_index=True)
            Py = pd.concat([Py, RNy], ignore_index=True)
            svc.fit(Dx, Py)

            # Save the first iteration
            # in case we need to rollback 
            # to this version
            if count == 0:
                self._best_model = deepcopy(svc)

            print("Iteration: {}".format(count))
            print("Unlabeled points remaining: {}\n".format(Ux.shape[0]))
            count += 1
                
            # Classify the rest of the unlabeled
            # points to extract more reliable negatives
            UL = svc.predict(Ux)
            indices = np.where(UL == 0)[0]
            # If no negative predictions are found,
            # we stop the loop
            if indices.size == 0:
                break
            else:
                # Extract new reliable negatives
                Qx = Ux.iloc[indices, :]
                Qy = Uy[indices]
                # Delete extracted items from the unlabeled set
                Ux = Ux.drop(indices)
                Uy = Uy.drop(indices)
                Ux = Ux.reset_index(drop=True)
                Uy = Uy.reset_index(drop=True)

            if Ux.empty:
                break
            
            # Merge previous reliable negatives with the 
            # newly acquired ones
            RNx = pd.concat([RNx, Qx], ignore_index=True)
            RNy = pd.concat([RNy, Qy], ignore_index=True)

        # Test the final classifier on the 
        # initial positive set
        predictions = svc.predict(Px)
        positive = len([x for x in predictions if x == 1])
        negative = len([x for x in predictions if x == 0])

        ratio = negative / positive
        print("Negative/Positive ratio: {}".format(ratio))
        # If the misclassified points are above 5% 
        # we will rollback to the cached model
        # Else we keep the final classifier
        if ratio < 0.05:
            self._best_model = svc   

        return Ux_test

    def predict(self, X):
        # Make a prediction on the whole unlabeled set
        # using the best classifier so far
        predictions = self._best_model.predict(X)
        # Check out how many got classified with 0 or 1
        results = np.unique(predictions, return_counts=True)
        results = np.array(results).T
        print(results)
        
        return predictions