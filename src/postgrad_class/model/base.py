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

    def score(self, X, y):
        """Optional: Return a default scoring metric (e.g., MSE)."""
        preds = self.predict(X)
        return ((preds - y) ** 2).mean()
