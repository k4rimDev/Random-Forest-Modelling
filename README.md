# Random Forest Package

A Python package for advanced Random Forest modeling, including classification and regression, hyperparameter tuning, and model visualization.

## Features

- **Random Forest Modeling**: Supports `RandomForestClassifier` and `RandomForestRegressor`.
- **Model Tuning**: Perform hyperparameter tuning using grid search and randomized search.
- **Model Evaluation**: Evaluate model performance with cross-validation.
- **Visualization**: Visualize model performance with confusion matrices, ROC curves, and precision-recall curves.
- **Custom Exceptions**: Handles errors with custom exception classes.

## Installation

You can install the package using pip:

```bash
pip install random-forest-package
```
## Usage

- Basic Example:
```py
from random_forest_package.model import RandomForestModel
from random_forest_package.tuner import ModelTuner
from random_forest_package.visualizer import ModelVisualizer

# Initialize and train the model
rf_model = RandomForestModel(n_estimators=100, random_state=42)
rf_model.train(X_train, y_train)

# Perform hyperparameter tuning
tuner = ModelTuner(rf_model)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
tuner.grid_search(X_train, y_train, param_grid)

# Visualize model performance
visualizer = ModelVisualizer(rf_model)
visualizer.plot_confusion_matrix(X_test, y_test)
visualizer.plot_roc_curve(X_test, y_test)
visualizer.plot_precision_recall_curve(X_test, y_test)
```

- Advanced Tuning Example:
```py
from random_forest_package.tuner import ModelTuner
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Initialize and tune the model
model = RandomForestClassifier()
tuner = ModelTuner(model)
param_distributions = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
tuner.randomized_search(X, y, param_distributions, n_iter=10)

# Cross-validation
results = tuner.cross_validate(X, y)
print(f"Mean score: {results['mean_score']}, Std score: {results['std_score']}")
```

Creating and Using a Random Forest Classifier

```py
from random_forest_package.classifier import RandomForestClassifierModel
from random_forest_package.trainer import ModelTrainer
from random_forest_package.evaluator import ModelEvaluator
from random_forest_package.tuner import ModelTuner
from random_forest_package.visualizer import ModelVisualizer

# Create a Random Forest Classifier
classifier = RandomForestClassifierModel(n_estimators=100, max_depth=10, random_state=42)

# Train the Classifier
trainer = ModelTrainer(classifier)
trainer.train(X_train, y_train)

# Evaluate the Classifier
evaluator = ModelEvaluator(classifier)
accuracy, conf_matrix, class_report = evaluator.evaluate(X_test, y_test)

# Tune the Classifier's Hyperparameters
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
tuner = ModelTuner(classifier, param_grid, search_type='grid')
best_params = tuner.tune(X_train, y_train)

# Visualize the Classifier's Performance
visualizer = ModelVisualizer(classifier)
visualizer.plot_confusion_matrix(X_test, y_test)
visualizer.plot_roc_curve(X_test, y_test)
visualizer.plot_precision_recall_curve(X_test, y_test)

print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

```
Creating and Using a Random Forest Regressor

```py
from random_forest_package.regressor import RandomForestRegressorModel
from random_forest_package.trainer import ModelTrainer
from random_forest_package.evaluator import ModelEvaluator
from random_forest_package.tuner import ModelTuner
from random_forest_package.visualizer import ModelVisualizer

# Create a Random Forest Regressor
regressor = RandomForestRegressorModel(n_estimators=100, max_depth=10, random_state=42)

# Train the Regressor
trainer = ModelTrainer(regressor)
trainer.train(X_train, y_train)

# Evaluate the Regressor
evaluator = ModelEvaluator(regressor)
mse = evaluator.evaluate(X_test, y_test)

# Tune the Regressor's Hyperparameters
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
tuner = ModelTuner(regressor, param_grid, search_type='random')
best_params = tuner.tune(X_train, y_train)

print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)

```

### Preprocessing Data
To preprocess data:

```py
import pandas as pd
from random_forest_package.preprocess import preprocess_data

# Example data
X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
y = pd.Series([0, 1, 0])

X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.2, random_state=42)
```

### Visualization
Visualization functions can be used to generate plots of model performance:
```py
from random_forest_package.visualizer import ModelVisualizer

# Initialize the visualizer
visualizer = ModelVisualizer(rf_model)

# Plot confusion matrix
visualizer.plot_confusion_matrix(X_test, y_test)

# Plot ROC curve
visualizer.plot_roc_curve(X_test, y_test)

# Plot precision-recall curve
visualizer.plot_precision_recall_curve(X_test, y_test)
```

## Custom Exceptions
This package provides custom exceptions for better error handling:

* `ModelCreationError`: Raised when there is an error creating the random forest model.
* `PreprocessingError`: Raised when there is an error during data preprocessing.
* `TrainingError`: Raised when there is an error during model training.
* `EvaluationError`: Raised when there is an error during model evaluation.
* `VisualizationError`: Raised when there is an error during visualization.


Example of handling a custom exception:


```py
class ModelCreationError(Exception):
    """Raised when there is an error in creating the model."""
    pass

class TrainingError(Exception):
    """Raised when there is an error during training."""
    pass

class EvaluationError(Exception):
    """Raised when there is an error during evaluation."""
    pass

```

## Testing
Tests are written using pytest. To run the tests:

```sh
poetry run pytest
```

## Project Structure


```
random_forest_package/
│
├── random_forest_package/
│   ├── __init__.py
│   ├── base_model.py          # Contains the abstract base class for the models
│   ├── classifier.py          # Contains the RandomForestClassifier class
│   ├── regressor.py           # Contains the RandomForestRegressor class
│   ├── preprocess.py          # Contains data preprocessing classes or functions
│   ├── trainer.py             # Contains classes for training models
│   ├── evaluator.py           # Contains classes for evaluating models
│   ├── utils.py               # Utility functions or classes
│   ├── visualizer.py          # Utility visualize cases
│   └── exceptions.py          # Custom exceptions
│
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py     # Tests for the classifier
│   ├── test_regressor.py      # Tests for the regressor
│   ├── test_preprocess.py     # Tests for preprocessing
│   ├── test_trainer.py        # Tests for training
│   ├── test_evaluator.py      # Tests for evaluation
│   └── test_utils.py          # Tests for utility functions
│
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml

```

## License
This project is licensed under the MIT License - see the [LICENSE]() file for details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
Karim Mirzaguliyev - [karimmirzaguliyev@gmail.com](mailto:karimmirzaguliyev@gmail.com)