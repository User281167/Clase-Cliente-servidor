import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# LINEAR
# ──────────────────────────────────────────────
# f(x) = x,   f'(x) = 1
#
class Linear(ActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)
