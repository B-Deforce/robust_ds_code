from dataclasses import dataclass
from omegaconf import MISSING # used by Hydra
import typing as tp

@dataclass
class LinearModelConfig:
    """Configuration for the linear model.
    This class is used to define the parameters for the linear model.

    Args:
        _target_ (str): The Hydra target class for the linear model.
    """
    _target_: str = "postgrad_class.model.linear.LinearModel"

@dataclass
class NeuralNetConfig:
    """Configuration for the neural network model.
    This class is used to define the parameters for the neural network model.

    Args:
        _target_ (str): The Hydra target class for the neural network model.
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the first layer.
        lr (float): Learning rate for the optimizer.
    """
    _target_: str = "postgrad_class.model.neural.SimpleNNModel"
    input_dim: int = MISSING
    hidden_dim: int = 32
    lr: float = 1e-3

@dataclass
class Config:
    """Main configuration class for the project.
    
    Args:
        model (Union[LinearModelConfig, NeuralNetConfig]): The model configuration.
    """
    model: tp.Any