from pulearn.pubase import *

import numpy as np


class ENClassifier(PUClassifier):
    """ 
        Classifier based on the first iteration of the Elkan Noto pape
    """

    def __init__(self, estimator) -> None:
        """ Constructor

        Args:
            estimator (sklearn.BaseEstimator): the model to use for the classification
            ratio (float, optional): the precentage of positive samples to leave unchanged. Defaults to 0.1.
        """

        super().__init__()

        self.estimator = estimator
        self.Ps1y1 = 0.0
    
    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def Ps1y1(self):
        return self._Ps1y1

    @Ps1y1.setter
    def Ps1y1(self, value):
        if value < 0.0 or value > 1.0:
            raise ValueError("[o_O] Probability values should be between 0-1")
        self._Ps1y1 = value

    def fit(self, X, y, ratio=0.2):
        """ Fit the classifier on the data

        Args:
            X (pandas.DataFrame): the data without any labels
            y (pandas.Series): the pu labels

        Returns:
            numpy float: the probability that a positive labeled example is actually positive
        """

        # Extract positive sample
        indices = extract_sample(y, ratio)

        X_out = X.iloc[indices, :]
        # Remove drawn sample
        y = y.drop(indices)
        X = X.drop(indices)
        # Fit the model to learn 
        # the probability that an 
        # element is labeled Pr(s=1|x)
        self.estimator.fit(X, y)

        # Predict the probability that the known positive samples are labeled
        predictions = self.estimator.predict_proba(X_out)[:, 1]    
        
        # Calculate the mean probability of the above predictions --> Pr(s=1|y=1)
        self.Ps1y1 = np.mean(predictions)

    def predict(self, X, threshold=0.5):
        """ Make the actual predictions

        Args:
            X (pandas.DataFrame): the data without labels
            threshold (float, optional): a bound to decide when an example is positive or negative. Defaults to 0.5.

        Returns:
            list: the final predictions
        """
        predicted_probs = self.estimator.predict_proba(X)[:, 1]

        try:
            predicted_probs /= self.Ps1y1
        except ZeroDivisionError:
            print("[o_O] Division by zero in pu_prob method --> Pr(s=1|y=1) = {}".format(self.Ps1y1))
            predicted_probs = 0

        predictions = [1 if prob > threshold else 0 for prob in predicted_probs] 

        return predictions


class WeightedENCLassifier(ENClassifier):
    """ 
        The second algorithm of the Elkan Noto paper
    """

    def __init__(self, estimator, labeled, unlabeled) -> None:
        super().__init__(estimator)
        
        self.labeled = labeled
        self.unlabeled = unlabeled

    def predict(self, X, threshold=0.5):
        """ Make the actual predictions

        Args:
            X (pandas.DataFrame): the data without labels
            threshold (float, optional): a bound to decide when an example is positive or negative. Defaults to 0.5.

        Returns:
            list: the final predictions
        """
        predicted_probs = self.estimator.predict_proba(X)[:, 1]

        np.place(predicted_probs, predicted_probs == 1.0, 0.999)
        weights = (predicted_probs / (1 - predicted_probs)) * ((1 - self.Ps1y1) / self.Ps1y1)

        numerator = float(self.labeled + weights.sum())
        denominator = float(self.labeled + self.unlabeled)  

        try:
            estimate = numerator / denominator      
            predicted_probs = predicted_probs * (self.Ps1y1 * estimate * (self.labeled + self.unlabeled))
            predicted_probs /= float(self.labeled)
        except ZeroDivisionError:
            print("[o_O] Division by zero in predict method")
            predicted_probs = 0

        predictions = [1 if prob > threshold else 0 for prob in predicted_probs] 

        return predictions