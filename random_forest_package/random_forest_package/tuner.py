# random_forest_package/tuner.py

import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

from random_forest_package.exceptions import TuningError


class ModelTuner:
    def __init__(self, model, scoring='accuracy', n_jobs=-1):
        self.model = model
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None

    def _handle_error(self, message):
        raise TuningError(message)
    
    def _extracted_from_randomized_search(self, arg0, X, y):
        arg0.fit(X, y)
        self.best_params_ = arg0.best_params_
        self.best_score_ = arg0.best_score_

    def grid_search(self, X, y, param_grid, cv=5):
        try:
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=1
            )
            self._extracted_from_randomized_search(grid_search, X, y)
        except Exception as e:
            self._handle_error(f"Error during grid search: {e}")

    def randomized_search(self, X, y, param_distributions, n_iter=10, cv=5):
        try:
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=1
            )
            self._extracted_from_randomized_search(random_search, X, y)
        except Exception as e:
            self._handle_error(f"Error during randomized search: {e}")

    def cross_validate(self, X, y, cv=5):
        try:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
        except Exception as e:
            self._handle_error(f"Error during cross-validation: {e}")
