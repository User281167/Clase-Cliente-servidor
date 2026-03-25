import numpy as np

from .cost_function import CostFunction


# ──────────────────────────────────────────────
# Binary Cross-Entropy
# ──────────────────────────────────────────────
#
#  Para K=2 o clasificación multi-label (cada clase independiente).
#
#  L = -(1/N) Σ_n [ y_n·log(ŷ_n) + (1-y_n)·log(1-ŷ_n) ]
#
#  ∂L/∂ŷ = (1/N)[ -y/ŷ + (1-y)/(1-ŷ) ]
#         = (1/N)[ (ŷ - y) / (ŷ(1-ŷ)) ]
#
#  Se usa con Sigmoid en la capa de salida (no Softmax).
#  Si la combinas con Sigmoid, el gradiente respecto a z colapsa a:
#    ∂L/∂z = (1/N)(ŷ - y)   ← mismo resultado que CE+Softmax
#
class BinaryCrossEntropy(CostFunction):
    """
    y_true : (N,) o (N, 1)  valores en {0, 1}
    y_pred : (N,) o (N, 1)  probabilidades en (0, 1)  — post Sigmoid
    """

    def function(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (1 / y_true.shape[0]) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))
