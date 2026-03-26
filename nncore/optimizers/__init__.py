from typing_extensions import Literal

from .adam import Adam
from .optimizer import Optimizer
from .rms_prop import RMSProp
from .sgd import SGD
from .sgd_momentum import SGDMomentum

OPTIMIZERS = {
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "rmsprop": RMSProp,
    "adam": Adam,
}

OptimizerName = Literal["sgd", "sgd_momentum", "rmsprop", "adam"]


def get_optimizer(name: OptimizerName, **kwargs) -> Optimizer:
    """
    get_optimizer("adam", learning_rate=0.001)
    get_optimizer("sgd_momentum", learning_rate=0.01, beta=0.9)
    """
    return OPTIMIZERS[name](**kwargs)


__all__ = ["Adam", "Optimizer", "RMSProp", "SGD", "SGDMomentum", "get_optimizer"]
