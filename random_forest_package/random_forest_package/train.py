from random_forest_package.exceptions import TrainingError


def train_model(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise TrainingError(f"Failed to train model: {e}")
