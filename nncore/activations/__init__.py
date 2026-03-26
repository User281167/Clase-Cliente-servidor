from typing import Literal

from .activation import ActivationFunction
from .elu import ELU
from .leaky_relu import LeakyReLU
from .linear import Linear
from .relu import ReLU
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh

ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "elu": ELU,
    "linear": Linear,
}

ActivationName = Literal[
    "relu", "leaky_relu", "sigmoid", "tanh", "softmax", "elu", "linear"
]


def get_activation(activation: ActivationName, **kwargs) -> ActivationFunction:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Activation {activation} not supported")
    return ACTIVATIONS[activation](**kwargs)


__all__ = [
    "ACTIVATIONS",
    "get_activation",
    "ReLU",
    "LeakyReLU",
    "Linear",
    "Sigmoid",
    "ELU",
    "Tanh",
    "Softmax",
    "ActivationFunction",
]
