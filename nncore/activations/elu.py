import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# ELU
# ──────────────────────────────────────────────
#         ⎧  x          si x > 0
# f(x) =  ⎨
#         ⎩  α(e^x - 1) si x ≤ 0      (α típico = 1.0)
#
# f'(x) = ⎧  1          si x > 0
#         ⎩  f(x) + α   si x ≤ 0
#
class ELU(ActivationFunction):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def function(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
