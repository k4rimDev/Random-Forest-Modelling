from sklearn.ensemble import RandomForestRegressor
from random_forest_package.base_model import RandomForestBaseModel
from random_forest_package.exceptions import ModelCreationError, TrainingError, EvaluationError
from sklearn.metrics import mean_squared_error


class RandomForestRegressorModel(RandomForestBaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        try:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        except Exception as e:
            raise ModelCreationError(f"Error creating RandomForestRegressor: {e}") from e

    def train(self, X, y):
        try:
            self.model.fit(X, y)
        except Exception as e:
            raise TrainingError(f"Error training RandomForestRegressor: {e}") from e

    def evaluate(self, X, y):
        try:
            predictions = self.model.predict(X)
            return mean_squared_error(y, predictions)
        except Exception as e:
            raise EvaluationError(f"Error evaluating RandomForestRegressor: {e}") from e
