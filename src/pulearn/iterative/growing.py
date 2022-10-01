from base import *


class GrowingClassifier(IterativeClassifier):
    def __init__(self, est_name1, est_name2, params1={}, params2={}) -> None:
        super().__init__(est_name1, est_name2, params1, params2)

    def _update_negatives(self, predictions):
        """ Update reliably negatives and shrink unlabeled set

        Args:
            predictions (nd.array): the partial predictions of the model of a specific iteration

        Returns:
            int: a status code
        """

        values = isolate_reliable_sample(self.Ux, self.Uy, predictions)
        if values is None:
            return 1
        
        # Extract new negative points and 
        # update remaining unlabeled ones
        self.Ux, self.Uy, Qx, Qy = values
        if self.Ux.empty:
            return 1
        
        # Merge previous reliable negatives with the 
        # newly acquired ones
        self.RNx = pd.concat([self.RNx, Qx], ignore_index=True)
        self.RNy = pd.concat([self.RNy, Qy], ignore_index=True)

        return 0

    def _predict_partial(self):
        """ Use classifier on unlabeled data

        Returns:
            nd.array: prediction results
        """
        return self._estimator.predict(self.Ux)