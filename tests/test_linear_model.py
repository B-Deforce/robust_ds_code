from postgrad_class.model.linear import LinearModel
import numpy as np
import pytest
from sklearn.datasets import make_regression

import typing as tp

@pytest.fixture
def linear_model() -> LinearModel:
    """Fixture to create a LinearModel instance."""
    return LinearModel()

@pytest.fixture
def regression_data() -> tp.Tuple[np.ndarray, np.ndarray]:
    """Fixture to create synthetic regression data."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    return X, y

@pytest.fixture
def regression_data_with_nan() -> tp.Tuple[np.ndarray, np.ndarray]:
    """Fixture to create synthetic regression data with NaN values."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X[0, 0] = np.nan
    return X, y

@pytest.fixture
def regression_data_with_inf() -> tp.Tuple[np.ndarray, np.ndarray]:
    """Fixture to create synthetic regression data with Inf values."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X[0, 0] = np.inf
    return X, y


def test_fit_runs(linear_model: LinearModel, regression_data: tp.Tuple[np.ndarray, np.ndarray]) -> None:
    """
    Test that the model can fit on clean regression data without errors.
    """
    X, y = regression_data
    linear_model.fit(X, y)