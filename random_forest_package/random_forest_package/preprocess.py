from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random_forest_package.exceptions import PreprocessingError


def preprocess_data(X, y, test_size=0.2, random_state=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise PreprocessingError(f"Failed to preprocess data: {e}")
