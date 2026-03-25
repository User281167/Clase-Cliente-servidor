import numpy as np

from .weight_initializer import WeightInitializer


# ──────────────────────────────────────────────
# Xavier / Glorot Uniform    [Glorot & Bengio, 2010]
# ──────────────────────────────────────────────
#
#  Diseñada para mantener varianza de activaciones constante
#  a través de capas con activaciones SIMÉTRICAS (Tanh, Sigmoid).
#
#  Derivación: si Var(x) = 1 y queremos Var(Wx) = 1, necesitamos:
#
#    Var(W_ij) = 2 / (n_in + n_out)
#
#  Para distribución uniforme U(-a, a), Var = a²/3, despejando a:
#
#    a = sqrt(6 / (n_in + n_out))
#
#  W ~ U(-a, a),   a = sqrt(6 / (n_in + n_out))
#
#  Usar con: Tanh, Sigmoid
#  No usar con: ReLU (ReLU rompe la simetría, usa He)
#
class XavierUniform(WeightInitializer):
    """
    W ~ U(-a, a)  donde  a = sqrt(6 / (n_in + n_out))
    Para activaciones simétricas: Tanh, Sigmoid.
    """

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))
