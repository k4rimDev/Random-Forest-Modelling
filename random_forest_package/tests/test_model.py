import pytest
from random_forest_package.model import create_random_forest
from random_forest_package.exceptions import ModelCreationError


def test_create_random_forest():
    model = create_random_forest()
    assert model is not None


def test_create_random_forest_exception():
    with pytest.raises(ModelCreationError):
        create_random_forest(n_estimators="invalid")
