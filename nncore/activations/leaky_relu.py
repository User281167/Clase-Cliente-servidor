import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# Leaky ReLU
# ──────────────────────────────────────────────
#
#          ⎧  α·x  si x ≤ 0      (α pequeño, típico 0.01)
#  f(x) =  ⎨
#          ⎩  x    si x > 0
#
#          ⎧  α    si x < 0
#  f'(x) = ⎨
#          ⎩  1    si x > 0
#
#  Soluciona el dying ReLU manteniendo gradiente no-nulo para x<0
#
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def function(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, self.alpha)
