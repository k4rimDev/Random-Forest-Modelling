# tests/test_tuner.py

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from random_forest_package.tuner import ModelTuner
from random_forest_package.exceptions import TuningError


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_grid_search(data):
    X, y = data
    model = RandomForestClassifier()
    tuner = ModelTuner(model)
    param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
    tuner.grid_search(X, y, param_grid)
    assert tuner.best_params_ is not None
    assert tuner.best_score_ is not None
    assert isinstance(tuner.best_params_, dict)


def test_randomized_search(data):
    X, y = data
    model = RandomForestClassifier()
    tuner = ModelTuner(model)
    param_distributions = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
    tuner.randomized_search(X, y, param_distributions, n_iter=5)
    assert tuner.best_params_ is not None
    assert tuner.best_score_ is not None
    assert isinstance(tuner.best_params_, dict)


def test_cross_validate(data):
    X, y = data
    model = RandomForestClassifier()
    tuner = ModelTuner(model)
    results = tuner.cross_validate(X, y)
    assert "mean_score" in results
    assert "std_score" in results
    assert isinstance(results["mean_score"], float)
    assert isinstance(results["std_score"], float)


def test_grid_search_error_handling(data):
    X, y = data
    model = RandomForestClassifier()
    tuner = ModelTuner(model)
    with pytest.raises(TuningError):
        # Provide invalid parameters to force an error
        tuner.grid_search(X, y, param_grid={"n_estimators": "invalid"})
