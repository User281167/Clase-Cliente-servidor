from .dense import Dense
from .dropout import Dropout
from .layer import Layer

LAYERS = {
    "dense": Dense,
    "dropout": Dropout,
}


def get_layer(name: str) -> Layer:
    if name not in LAYERS:
        raise ValueError(
            f"Unknown layer: {name}, available layers: {list(LAYERS.keys())}"
        )
    return LAYERS[name]


__all__ = ["get_layer", "Layer", "Dense", "Dropout"]
