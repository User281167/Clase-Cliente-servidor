import numpy as np

from .weight_initializer import WeightInitializer


# ──────────────────────────────────────────────
# Xavier / Glorot Normal
# ──────────────────────────────────────────────
#
#  Misma motivación que XavierUniform pero usando gaussiana.
#
#  Para N(0, σ²), Var = σ², despejando:
#
#    σ = sqrt(2 / (n_in + n_out))
#
#  W ~ N(0, σ²),   σ = sqrt(2 / (n_in + n_out))
#
#  En la práctica XavierUniform y XavierNormal dan resultados similares.
#  Normal es levemente mejor cuando la red es muy profunda.
#
class XavierNormal(WeightInitializer):
    """
    W ~ N(0, σ²)  donde  σ = sqrt(2 / (n_in + n_out))
    Para activaciones simétricas: Tanh, Sigmoid.
    """

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        std = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std
