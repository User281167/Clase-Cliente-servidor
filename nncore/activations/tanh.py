import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# Tanh  —  tangente hiperbólica
# ──────────────────────────────────────────────
#
#          e^x - e^(-x)
#  f(x) = ──────────────    rango: (-1, 1)
#          e^x + e^(-x)
#
#  f'(x) = 1 - tanh²(x) = 1 - f(x)²
#
#  Ventaja sobre Sigmoid: centrada en 0 → gradientes más simétricos
#  Misma debilidad: saturación en |x| grande
#
class Tanh(ActivationFunction):
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2  # más estable que reusar self.function
