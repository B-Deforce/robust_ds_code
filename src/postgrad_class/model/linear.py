from sklearn.linear_model import LinearRegression
import numpy as np
from beartype import beartype
from postgrad_class.model.base import BaseModel


@beartype
class LinearModel(BaseModel):
    """Linear regression model using scikit-learn's LinearRegression.
    This model is suitable for regression tasks where the relationship between the input features and the target variable is linear.
    It uses the LinearRegression class from scikit-learn to fit the model to the training data and make predictions on new data.

    Args:
        BaseModel (class): The base class for all models in the postgrad_class package.
    """
    def __init__(self):
        """Initializes the LinearModel class."""
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input features using the trained model.
        Args:
            X (np.ndarray): The input features for which to make predictions.
        Returns:
            np.ndarray: The predicted target variable.
        """
        return self.model.predict(X)