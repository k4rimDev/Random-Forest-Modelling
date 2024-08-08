from abc import ABC, abstractmethod


class RandomForestBaseModel(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass
