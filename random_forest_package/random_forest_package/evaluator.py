class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)
