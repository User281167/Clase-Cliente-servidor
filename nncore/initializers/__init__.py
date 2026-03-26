from typing import Literal

from .he_normal import HeNormal
from .he_uniform import HeUniform
from .random_normal import RandomNormal
from .weight_initializer import WeightInitializer
from .xavier_normal import XavierNormal
from .xavier_uniform import XavierUniform

INITIALIZERS = {
    "random_normal": RandomNormal,
    "xavier_uniform": XavierUniform,
    "xavier_normal": XavierNormal,
    "he_normal": HeNormal,
    "he_uniform": HeUniform,
}

InitializerName = Literal[
    "random_normal", "xavier_uniform", "xavier_normal", "he_normal", "he_uniform"
]


def get_initializer(name: InitializerName, **kwargs) -> WeightInitializer:
    if name not in INITIALIZERS:
        raise ValueError(
            f"Unknown initializer: {name}, available initializers: {list(INITIALIZERS.keys())}"
        )
    return INITIALIZERS[name](**kwargs)


__all__ = [
    "INITIALIZERS",
    "get_initializer",
    "WeightInitializer",
    "HeNormal",
    "HeUniform",
    "RandomNormal",
    "XavierNormal",
    "XavierUniform",
]
