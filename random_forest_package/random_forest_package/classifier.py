from sklearn.ensemble import RandomForestClassifier
from random_forest_package.base_model import RandomForestBaseModel
from random_forest_package.exceptions import ModelCreationError, TrainingError, EvaluationError
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class RandomForestClassifierModel(RandomForestBaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        try:
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        except Exception as e:
            raise ModelCreationError(f"Error creating RandomForestClassifier: {e}") from e

    def train(self, X, y):
        try:
            self.model.fit(X, y)
        except Exception as e:
            raise TrainingError(f"Error training RandomForestClassifier: {e}") from e

    def evaluate(self, X, y):
        try:
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            conf_matrix = confusion_matrix(y, predictions)
            class_report = classification_report(y, predictions)
            return accuracy, conf_matrix, class_report
        except Exception as e:
            raise EvaluationError(f"Error evaluating RandomForestClassifier: {e}") from e
