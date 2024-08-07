class RandomForestPackageError(Exception):
    """Base class for other exceptions in this package."""
    pass


class ModelCreationError(RandomForestPackageError):
    """Raised when there is an error creating the random forest model."""
    def __init__(self, message="Error creating the random forest model"):
        self.message = message
        super().__init__(self.message)


class PreprocessingError(RandomForestPackageError):
    """Raised when there is an error during data preprocessing."""
    def __init__(self, message="Error during data preprocessing"):
        self.message = message
        super().__init__(self.message)


class TrainingError(RandomForestPackageError):
    """Raised when there is an error during model training."""
    def __init__(self, message="Error during model training"):
        self.message = message
        super().__init__(self.message)


class EvaluationError(RandomForestPackageError):
    """Raised when there is an error during model evaluation."""
    def __init__(self, message="Error during model evaluation"):
        self.message = message
        super().__init__(self.message)
