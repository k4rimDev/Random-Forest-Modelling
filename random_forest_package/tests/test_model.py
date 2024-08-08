import pytest

from random_forest_package.exceptions import ModelCreationError
from random_forest_package.model import (
    create_random_forest_classifier,
    create_random_forest_regressor
)


def test_create_random_forest_classifier():
    model = create_random_forest_classifier()
    assert model is not None


def test_create_random_forest_regressor():
    model = create_random_forest_regressor()
    assert model is not None


def test_create_random_forest_classifier_exception():
    with pytest.raises(ModelCreationError):
        create_random_forest_classifier(n_estimators="invalid")

def test_create_random_forest_regressor_exception():
    with pytest.raises(ModelCreationError):
        create_random_forest_regressor(n_estimators="invalid")
