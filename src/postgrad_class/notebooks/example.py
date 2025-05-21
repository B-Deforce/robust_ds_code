# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
from dataclasses import dataclass, field
import pandas as pd
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# %%
X, y = make_regression()

# %% [markdown]
# # 1. Designing Clean Code with Reusable Structure

# %% [markdown]
# ### Imperative

# %%
# no structure, hard to reuse/maintain
model = LinearRegression()

model.fit(X, y)

preds = model.predict(X)


# %% [markdown]
# What if I want to:
# * store predictions as an attribute
# * add option to save the model in predict method
# * ...

# %% [markdown]
# ### OOP

# %%
# easier to maintain/extend and read
class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# %%
model = LinearModel()

model.fit(X, y)

preds = model.predict(X)


# %%
@dataclass
class ModelPrediction:
  """
  A class to hold model predictions and input features.
  It includes a method to save the predictions to a CSV file.
  Args:
    predictions (np.ndarray): The model predictions.
    input_features (np.ndarray): The input features used for predictions.
    timestamp (datetime): The timestamp of when the predictions were made.
  """
  predictions: np.ndarray
  input_features: np.ndarray
  timestamp: datetime = field(default_factory=datetime.now)

  def to_csv(self, results_dir: str) -> None:
      """
      Save the predictions and input features to a CSV file.
      Args:
        results_dir (str): The directory where the CSV file will be saved.
      """
      os.makedirs(results_dir, exist_ok=True)
      df = pd.DataFrame(self.predictions)
      df.to_csv(
        os.path.join(results_dir, f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_predictions.csv"), 
        index=False
      )


# %%
class LinearModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        preds = self.model.predict(X)
        return ModelPrediction(
          predictions=preds, 
          input_features=X
        )


# %%
preds.to_csv(results_dir="my_results")


# %% [markdown]
# #### Simple NN
# One can now easily swap out the `LinearModel` above.

# %%
class SimpleNNModel:
    def __init__(self, input_dim, hidden_dim=16, lr=1e-3):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y, epochs=100, verbose=True):
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

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X).numpy().flatten()
        self.predictions = preds
        return preds


# %% [markdown]
# ##### What if someone:

# %% [markdown]
# * does not follow our structure and uses a different naming convention?
#
# ![image](images/derp.png)

# %% [markdown]
# ### OOP with ABC

# %%
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        """Train the model using input features X and target y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for input features X."""
        pass

    @abstractmethod
    def score(self, X, y):
        """Optional: Return a default scoring metric (e.g., MSE)."""
        preds = self.predict(X)
        return ((preds - y) ** 2).mean()



# %%
class BaseResults(ABC):
    @abstractmethod
    def to_csv(self, results_dir: str) -> None:
        """Save the predictions and input features to a CSV file."""
        pass

    @abstractmethod
    def from_csv(self, csv_path: str) -> None:
        """Load predictions and input features from a CSV file."""
        pass


# %%
class BaseEvaluation(ABC):
    @abstractmethod
    def evaluate(self, model: BaseModel, X, y) -> dict:
        """Evaluate the model and return a dictionary of metrics."""
        pass


# %% [markdown]
# # 2. Readable, Safe, and Maintainable Code

# %% [markdown]
# What if someone:
# - passes the wrong type?
# - uses the neural network assuming the input should be a tensor?

# %% [markdown]
# ### Beartype

# %%
from beartype import beartype
#from jaxtyping import Num

# %% [markdown]
# What happens if we instead pass a pytorch tensor to our fit method?

# %%
@beartype
class SimpleNNModel:
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


# %%
model = SimpleNNModel(input_dim=X.shape[1], hidden_dim=16, lr=1e-3)

model.fit(X, y)

preds = model.predict(X)

# %% [markdown]
# Also see example of `ModelPrediction` as expected return type.
