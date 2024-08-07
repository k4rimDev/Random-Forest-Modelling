# Random Forest Package

A Python package to facilitate random forest modeling. This package provides functionalities for creating, training, preprocessing, and evaluating random forest models with custom exception handling.

## Installation

First, ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed. Then, you can install the package and its dependencies using:

```sh
poetry install
```
### Usage

Creating a Random Forest Model
To create a random forest model:

```py
from random_forest_package.model import create_random_forest

model = create_random_forest(n_estimators=100, max_depth=None, random_state=42)
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

### Training the Model
To train the random forest model:


```py
from random_forest_package.train import train_model

trained_model = train_model(model, X_train, y_train)
```

### Evaluating the Model
To evaluate the model:

```py
from random_forest_package.evaluate import evaluate_model

accuracy, conf_matrix, class_report = evaluate_model(trained_model, X_test, y_test)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Classification Report: \n{class_report}')
```

## Custom Exceptions
This package provides custom exceptions for better error handling:

* `ModelCreationError`: Raised when there is an error creating the random forest model.
* `PreprocessingError`: Raised when there is an error during data preprocessing.
* `TrainingError`: Raised when there is an error during model training.
* `EvaluationError`: Raised when there is an error during model evaluation.

Example of handling a custom exception:


```py
from random_forest_package.exceptions import ModelCreationError

try:
    model = create_random_forest(n_estimators="invalid")
except ModelCreationError as e:
    print(e)
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
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── exceptions.py
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_utils.py
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