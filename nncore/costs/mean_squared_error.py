import numpy as np

from .cost_function import CostFunction


# ──────────────────────────────────────────────
# Mean Squared Error
# ──────────────────────────────────────────────
#
#  L = (1/N) Σ_n ||y_n - ŷ_n||²
#    = (1/N) Σ_n Σ_k (y_nk - ŷ_nk)²
#
#  ∂L/∂ŷ = (2/N)(ŷ - y)
#
#  Para clasificación con one-hot, MSE penaliza todas las clases por igual
#  lo que no es ideal — CrossEntropy es mejor para clasificación.
#  MSE sí tiene sentido en la capa de salida de un autoencoder o regresión.
#
class MeanSquaredError(CostFunction):
    """
    y_true : (N, K) one-hot o valores continuos
    y_pred : (N, K) salida de la red (post-activación)
    """

    def function(self, y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    def derivative(self, y_true, y_pred):
        """
        ∂L/∂ŷ = (2/N)(ŷ - y)

        Factor 2 se suele absorber en el learning rate,
        algunos frameworks usan (1/N)(ŷ - y) omitiendo el 2.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]
