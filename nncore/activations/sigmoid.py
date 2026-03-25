import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# Sigmoid  (logística)
# ──────────────────────────────────────────────
#
#              1
#  f(x) = ──────────    rango: (0, 1)
#          1 + e^(-x)
#
#  f'(x) = f(x) · (1 - f(x))
#
#  Derivación:
#    Sea s = f(x). Entonces ds/dx = s(1-s)
#    (útil porque se puede reusar el forward pass)
#
#  Problema: saturación — para |x| grande, f'(x) ≈ 0 → vanishing gradient
#
class Sigmoid(ActivationFunction):
    def function(self, x):
        # Estabilidad numérica: evitar overflow en exp(-x) para x muy negativo
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def derivative(self, x):
        s = self.function(x)
        return s * (1 - s)
