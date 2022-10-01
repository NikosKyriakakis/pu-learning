from pulearn.iterative.base import *


# Do not use -- currently contains a bug
class PruningClassifier(IterativeClassifier):
    def __init__(
            self, 
            est_name1, 
            est_name2,
            params1={}, 
            params2={}     
        ) -> None:

        super().__init__(est_name1, est_name2, params1, params2)

    def _update_negatives(self, predictions):
        """ Update reliably negatives and shrink unlabeled set

        Args:
            predictions (nd.array): the partial predictions of the model of a specific iteration

        Returns:
            int: a status code
        """

        values = isolate_reliable_sample(self.RNx, self.RNy, predictions)
        if values is None:
            return 1

        _, _, Qx, Qy = values

        # print("RNx = {}".format(self.RNx.shape[0]))
        # print("Qx = {}".format(Qx.shape[0]))
        # print("Px = {}".format(self.Px.shape[0]))

        if (self._prev_Q.shape[0] > Qx.shape[0]) or (self.Px.shape[0] > self.RNx.shape[0]):
            return 1

        self._prev_Q = self.RNx = Qx.copy()
        self.RNy = Qy.copy()

        return 0

    def _predict_partial(self):
        return self._estimator.predict(self.RNx)