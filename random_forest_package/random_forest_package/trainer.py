class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.train(X, y)
