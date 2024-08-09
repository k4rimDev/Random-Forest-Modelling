import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

from random_forest_package.visualizer import ModelVisualizer
from random_forest_package.exceptions import VisualizationError


# Fixture to create a simple classification dataset
@pytest.fixture(scope='module')
def classification_data():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Fixture to create a trained RandomForestClassifierModel
@pytest.fixture(scope='module')
def trained_classifier(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Tests for plot_confusion_matrix
def test_plot_confusion_matrix_normal(trained_classifier):
    model, X_test, y_test = trained_classifier
    visualizer = ModelVisualizer(model)

    try:
        visualizer.plot_confusion_matrix(X_test, y_test)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_plot_confusion_matrix_with_normalization(trained_classifier):
    model, X_test, y_test = trained_classifier
    visualizer = ModelVisualizer(model)

    try:
        visualizer.plot_confusion_matrix(X_test, y_test, normalize=True)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_plot_confusion_matrix_with_invalid_input(trained_classifier):
    model, _, _ = trained_classifier
    visualizer = ModelVisualizer(model)

    with pytest.raises(VisualizationError):
        visualizer.plot_confusion_matrix(None, None)


# Tests for plot_roc_curve
def test_plot_roc_curve_normal(trained_classifier):
    model, X_test, y_test = trained_classifier
    visualizer = ModelVisualizer(model)

    try:
        visualizer.plot_roc_curve(X_test, y_test)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_plot_roc_curve_with_invalid_input(trained_classifier):
    model, _, _ = trained_classifier
    visualizer = ModelVisualizer(model)

    with pytest.raises(VisualizationError):
        visualizer.plot_roc_curve(None, None)


# Tests for plot_precision_recall_curve
def test_plot_precision_recall_curve_normal(trained_classifier):
    model, X_test, y_test = trained_classifier
    visualizer = ModelVisualizer(model)

    try:
        visualizer.plot_precision_recall_curve(X_test, y_test)
        plt.close()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_plot_precision_recall_curve_with_invalid_input(trained_classifier):
    model, _, _ = trained_classifier
    visualizer = ModelVisualizer(model)

    with pytest.raises(VisualizationError):
        visualizer.plot_precision_recall_curve(None, None)


def test_plot_precision_recall_curve_with_single_class(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    y_train_single_class = np.zeros_like(y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train_single_class)

    visualizer = ModelVisualizer(model)

    try:
        visualizer.plot_precision_recall_curve(X_test, y_test)
        plt.close()
    except VisualizationError:
        pass  # Expected outcome
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
