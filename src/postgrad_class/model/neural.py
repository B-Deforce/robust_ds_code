import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from beartype import beartype
from postgrad_class.model.base import BaseModel


@beartype
class SimpleNNModel(BaseModel):
    """A simple feedforward neural network model for regression tasks.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int, optional): Number of hidden units in the first layer. Default is 16.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = 16, lr: float | None = 1e-3):
        """Initialize the model, loss function, and optimizer."""
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int | None = 100, verbose: bool | None = True) -> None:
        """Train the model using input features X and target y.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target variable.
            epochs (int, optional): Number of training epochs. Default is 100.
            verbose (bool, optional): If True, print loss every 10 epochs. Default is True.
        Returns:
            None
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.model.train()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features X.
        
        Args:
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Predicted values.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X).numpy().flatten()
        self.predictions = preds
        return preds