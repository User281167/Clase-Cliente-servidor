from abc import ABC

import numpy as np

from nncore.activations import ActivationFunction
from nncore.initializers import WeightInitializer


class Layer(ABC):
    """
    Clase base para capas de una red neuronal.

    Una capa transforma una entrada X de shape (N, n_in)
    a una salida de shape (N, n_out) donde N = batch size.

    Todo Layer debe implementar:
      forward(X)      : paso hacia adelante  X -> ŷ
      backward(delta) : paso hacia atrás     ∂L/∂X <- ∂L/∂ŷ

    El backward recibe el gradiente de la loss respecto a la
    SALIDA de esta capa, y devuelve el gradiente respecto a
    la ENTRADA (para propagarlo a la capa anterior).
    Los gradientes de W y b se almacenan en self.d_weights, self.d_bias.
    """

    def __init__(
        self,
        input_size: int | None = None,
        output_size: int | None = None,
        activation: ActivationFunction | None = None,
        weight_initializer: WeightInitializer | None = None,
    ):
        # weight_initializer debe ser una INSTANCIA, no la clase
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.initializer = weight_initializer
        self.weights = (
            weight_initializer.initialize(input_size, output_size)
            if (
                weight_initializer is not None
                and output_size is not None
                and input_size is not None
            )
            else None
        )
        self.bias = np.zeros((1, output_size)) if output_size is not None else None

        # Gradientes — se llenan en backward, se leen en el optimizer
        self.d_weights = None
        self.d_bias = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in={self.input_size}, "
            f"out={self.output_size}, "
            f"activation={self.activation.__class__.__name__})"
        )
