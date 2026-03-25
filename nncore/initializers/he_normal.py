import numpy as np

from .weight_initializer import WeightInitializer


# ──────────────────────────────────────────────
# He Normal    [He et al., 2015]
# ──────────────────────────────────────────────
#
#  Xavier asume activaciones simétricas con media 0.
#  ReLU anula la mitad de las neuronas → efectivamente reduce
#  la varianza a la mitad.
#
#    Var(W_ij) = 2 / n_in
#
#  Para N(0, σ²):
#
#    σ = sqrt(2 / n_in)
#
#  W ~ N(0, σ²),   σ = sqrt(2 / n_in)
#
#  Usar con: ReLU, LeakyReLU, ELU
#  No usar con: Tanh, Sigmoid (usa Xavier)
#
class HeNormal(WeightInitializer):
    """
    W ~ N(0, σ²)  donde  σ = sqrt(2 / n_in)
    Para activaciones tipo ReLU.
    """

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        std = np.sqrt(2.0 / input_size)
        return np.random.randn(input_size, output_size) * std
