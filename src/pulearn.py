import numpy as np


class PULearner:
    """
        A base class which embeddes the core PU learning functionality
    """
    def __init__(self, estimator, ratio=0.1) -> None:
        """ Constructor

        Args:
            estimator (sklearn.BaseEstimator): the model to use for the classification
            ratio (float, optional): the precentage of positive samples to leave unchanged. Defaults to 0.1.
        """
        self.estimator = estimator
        self.ratio = ratio

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

    def extract_pos_sample(self, target):
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
            data (pandas.DataFrame): the original dataset
        """

        # Extract positive sample
        indices = self.extract_pos_sample(y)
        X_out = X.iloc[indices, :]

        # Remove drawn sample
        X = X.drop(indices)
        y = y.drop(indices)

        # Fit the model to learn 
        # the probability that an 
        # element is labeled Pr(s=1|x)
        self.estimator.fit(X, y)
        # 
        predictions = self.estimator.predict_proba(X_out)[:, 1]
    
        # Return the Pr(s=1|y=1)
        return np.mean(predictions)

    def pu_prob(self, X, Ps1y1):
        predicted_labels = self.estimator.predict_proba(X)[:, 1]
        return predicted_labels / Ps1y1

    



        
        