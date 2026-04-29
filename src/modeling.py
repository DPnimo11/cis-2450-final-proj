from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE


def chronological_train_test_split(X, y, test_size: float = TEST_SIZE):
    """Split without shuffling to reduce time leakage."""
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def scale_train_test(X_train, X_test):
    """Fit a StandardScaler on train only, then transform train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def fit_baseline_logistic_regression(X_train, y_train):
    """Train the current baseline Logistic Regression model."""
    model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model
