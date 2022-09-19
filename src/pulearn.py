class PULearner:
    """
        A base class which embeddes the core PU learning functionality
    """
    def __init__(self, estimator, leave_out=0.1) -> None:
        """ Constructor

        Args:
            estimator (sklearn.BaseEstimator): the model to use for the classification
            leave_out (float, optional): _description_. Defaults to 0.1.
        """
        self.estimator = estimator
        self.leave_out = leave_out

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def leave_out(self):
        return self._leave_out

    @leave_out.setter
    def leave_out(self, value):
        if type(value) != float:
            self._leave_out = 0.1
            return

        if value < 0.0 or value > 1.0:
            self._leave_out = 0.1
        else:
            self._leave_out = value
        