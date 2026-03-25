import numpy as np

from .weight_initializer import WeightInitializer


# ──────────────────────────────────────────────
# Random Normal
# ──────────────────────────────────────────────
#
#  W ~ N(0, 1)
#
#  El problema: sin escalar, la varianza de la salida crece con input_size.
#  Si input_size=784 (MNIST), las activaciones explotan inmediatamente.
#
# Nota:
#  No usar en producción — sirve para demostrar por qué importa
#  la inicialización correcta (experimento didáctico).
#
class RandomNormal(WeightInitializer):
    """
    W ~ N(0, 1)  — sin escalar.
    Solo útil para ilustrar el problema de inicialización naive.
    """

    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.randn(input_size, output_size) * self.std
