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
