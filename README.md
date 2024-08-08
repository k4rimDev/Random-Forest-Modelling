# Random Forest Package

A Python package to facilitate random forest modeling, supporting both classification and regression tasks using object-oriented design principles.

## Installation

To install the package, use:

```sh
pip install random-forest-package
```
### Usage

Creating and Using a Random Forest Classifier

```py
from random_forest_package.classifier import RandomForestClassifierModel
from random_forest_package.trainer import ModelTrainer
from random_forest_package.evaluator import ModelEvaluator

# Create a Random Forest Classifier
classifier = RandomForestClassifierModel(n_estimators=100, max_depth=10, random_state=42)

# Train the Classifier
trainer = ModelTrainer(classifier)
trainer.train(X_train, y_train)

# Evaluate the Classifier
evaluator = ModelEvaluator(classifier)
accuracy, conf_matrix, class_report = evaluator.evaluate(X_test, y_test)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

```
Creating and Using a Random Forest Regressor

```py
from random_forest_package.regressor import RandomForestRegressorModel
from random_forest_package.trainer import ModelTrainer
from random_forest_package.evaluator import ModelEvaluator

# Create a Random Forest Regressor
regressor = RandomForestRegressorModel(n_estimators=100, max_depth=10, random_state=42)

# Train the Regressor
trainer = ModelTrainer(regressor)
trainer.train(X_train, y_train)

# Evaluate the Regressor
evaluator = ModelEvaluator(regressor)
mse = evaluator.evaluate(X_test, y_test)

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

## Custom Exceptions
This package provides custom exceptions for better error handling:

* `ModelCreationError`: Raised when there is an error creating the random forest model.
* `PreprocessingError`: Raised when there is an error during data preprocessing.
* `TrainingError`: Raised when there is an error during model training.
* `EvaluationError`: Raised when there is an error during model evaluation.

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