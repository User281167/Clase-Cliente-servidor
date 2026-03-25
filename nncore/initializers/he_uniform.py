import numpy as np

from .weight_initializer import WeightInitializer


# ──────────────────────────────────────────────
# He Uniform
# ──────────────────────────────────────────────
#
#  Misma motivación que HeNormal, versión uniforme.
#  Para U(-a, a), Var = a²/3, despejando:
#
#    a = sqrt(6 / n_in)
#
#  W ~ U(-a, a),   a = sqrt(6 / n_in)
#
#  HeNormal es más común en la literatura.
#  HeUniform puede dar mejor resultado en redes poco profundas.
#
class HeUniform(WeightInitializer):
    """
    W ~ U(-a, a)  donde  a = sqrt(6 / n_in)
    Para activaciones tipo ReLU.
    """

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        limit = np.sqrt(6.0 / input_size)
        return np.random.uniform(-limit, limit, (input_size, output_size))
