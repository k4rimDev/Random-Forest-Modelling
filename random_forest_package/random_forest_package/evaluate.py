from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from random_forest_package.exceptions import EvaluationError


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, conf_matrix, class_report
    except Exception as e:
        raise EvaluationError(f"Failed to evaluate model: {e}")
