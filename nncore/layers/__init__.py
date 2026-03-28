from typing import Literal

from .conv2d import Conv2D
from .dense import Dense
from .dropout import Dropout
from .flatten import Flatten
from .layer import Layer
from .max_pool2d import MaxPool2D

LAYERS = {
    "dense": Dense,
    "dropout": Dropout,
    "flatten": Flatten,
    "max_pool2d": MaxPool2D,
    "conv2d": Conv2D,
}

LayerName = Literal["dense", "dropout", "flatten", "max_pool2d", "conv2d"]


def get_layer(name: LayerName, **kwargs) -> Layer:
    if name not in LAYERS:
        raise ValueError(
            f"Unknown layer: {name}, available layers: {list(LAYERS.keys())}"
        )
    return LAYERS[name](**kwargs)


__all__ = ["get_layer", "Layer", "Dense", "Dropout", "Flatten", "MaxPool2D", "Conv2D"]
