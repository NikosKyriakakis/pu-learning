import numpy as np

from abc import ABC, abstractmethod

class PUClassifier(ABC):
    def __init__(self) -> None:
        """ Base PU learning class """

    @abstractmethod
    def fit(self, X, y):
        """ Override to specify training behaviour """

    @abstractmethod
    def predict(self, X):
        """ Override to specify testing behaviour """


class ENClassifier(PUClassifier):
    """ 
        Classifier based on the first iteration of the Elkan Noto pape
    """

    def __init__(self, estimator, ratio=0.1) -> None:
        """ Constructor

        Args:
            estimator (sklearn.BaseEstimator): the model to use for the classification
            ratio (float, optional): the precentage of positive samples to leave unchanged. Defaults to 0.1.
        """

        super().__init__()

        self.estimator = estimator
        self.ratio = ratio
        self.Ps1y1 = 0.0
    
    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if type(value) != float:
            self._ratio = 0.1
            return

        if value < 0.0 or value > 1.0:
            self._ratio = 0.1
        else:
            self._ratio = value

    @property
    def Ps1y1(self):
        return self._Ps1y1

    @Ps1y1.setter
    def Ps1y1(self, value):
        if value < 0.0 or value > 1.0:
            raise ValueError("[o_O] Probability values should be between 0-1")
        self._Ps1y1 = value
            
    def extract_sample(self, target):
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
        threshold = int(np.ceil(self.ratio * len(positive_indices)))
        sample = positive_indices[:threshold]

        return sample

    def fit(self, X, y):
        """ Fit the classifier on the data

        Args:
            X (pandas.DataFrame): the data without any labels
            y (pandas.Series): the pu labels

        Returns:
            numpy float: the probability that a positive labeled example is actually positive
        """

        # Extract positive sample
        indices = self.extract_sample(y)
        X_out = X.iloc[indices, :]

        # Remove drawn sample
        X = X.drop(indices)
        y = y.drop(indices)

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