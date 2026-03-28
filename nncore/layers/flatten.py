import numpy as np

from .layer import Layer

# ──────────────────────────────────────────────
# Flatten  —  capa de aplanamiento
# ──────────────────────────────────────────────
#
#  Convierte un tensor multidimensional en un vector por muestra.
#  Se usa tipicamente entre la ultima capa convolucional/pooling
#  y la primera capa Dense.
#
#  No tiene parametros aprendibles (sin W, sin b, sin activacion).
#
#  FORWARD
#  ───────
#  Entrada X de shape (N, C, H, W)  ← tipico output de Conv/Pool
#                  o (N, ...)        ← cualquier shape arbitraria
#
#    A = X.reshape(N, -1)            shape: (N, C*H*W)
#
#  Se guarda X.shape para poder revertirlo en el backward.
#
#  BACKWARD
#  ────────
#  Recibe delta = ∂L/∂A de shape (N, C*H*W)
#  No hay pesos ni activacion — solo revertir el reshape:
#
#    ∂L/∂X = delta.reshape(input_shape)   shape: (N, C, H, W)
#
#  El gradiente fluye hacia atras sin transformacion algebraica,
#  unicamente recuperando la forma original del tensor.


class Flatten(Layer):
    """
    Capa de aplanamiento sin parametros aprendibles.

    Convierte un tensor (N, C, H, W) — o cualquier shape (N, ...) —
    en una matriz 2D (N, C*H*W) para conectar capas convolucionales
    con capas Dense.

    Atributos guardados en forward
    ──────────────────────────────
    _input_shape : tuple
        Shape completo de X, necesario para revertir el reshape
        en el backward pass.
    """

    def __init__(self):
        # Sin input_size/output_size/activation/initializer —
        # se infieren dinamicamente en el primer forward.
        super().__init__()
        self._input_shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._input_shape = X.shape  # guarda (N, C, H, W) para backward
        return X.reshape(X.shape[0], -1)  # (N, C*H*W)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta.reshape(self._input_shape)  # (N, C, H, W)

    def __repr__(self):
        shape = self._input_shape
        flat = int(np.prod(shape[1:])) if shape is not None else "?"
        return f"Flatten(in={shape[1:] if shape else '?'}, out={flat})"

    def compute_output_shape(self, input_shape):
        return (int(np.prod(input_shape)),)
