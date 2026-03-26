from typing import Literal

from .dense import Dense
from .dropout import Dropout
from .layer import Layer

LAYERS = {
    "dense": Dense,
    "dropout": Dropout,
}

LayerName = Literal["dense", "dropout"]


def get_layer(name: LayerName, **kwargs) -> Layer:
    if name not in LAYERS:
        raise ValueError(
            f"Unknown layer: {name}, available layers: {list(LAYERS.keys())}"
        )
    return LAYERS[name](**kwargs)


__all__ = ["get_layer", "Layer", "Dense", "Dropout"]
