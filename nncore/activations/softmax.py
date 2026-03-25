import numpy as np

from .activation import ActivationFunction


# ──────────────────────────────────────────────
# Softmax
# ──────────────────────────────────────────────
#
#           e^(x_i - max(x))
#  f(x)_i = ─────────────────      i = 0,...,K-1
#            Σ_j e^(x_j - max(x))
#
#  El shift por max(x) es estabilidad numérica (evita overflow), no cambia el resultado:
#    e^(x_i - c) / Σ e^(x_j - c) = e^x_i / Σ e^x_j
#
#  Salida: distribución de probabilidad — Σ_i f(x)_i = 1, f(x)_i ∈ (0,1)
#
#  Jacobiana (derivada completa es una matriz, no un vector):
#
#    ∂f_i/∂x_j = f_i · (δ_ij - f_j)
#
#    En forma matricial para un vector s = softmax(x):
#      J = diag(s) - s·sᵀ     (matriz K×K)
#
#  NOTA:
#  En la práctica, Softmax se usa SIEMPRE con Cross-Entropy como loss.
#  El gradiente combinado ∂L/∂x = ŷ - y es muy simple (ver CostFunction).
#  La Jacobiana completa RARAMENTE se necesita sola.
#
#  El método derivative() aquí devuelve el jacobiano por fila (útil para
#  visualización o casos edge), PERO en el backward pass de la red
#  usar el gradiente combinado Softmax + CrossEntropy directamente.
#
class Softmax(ActivationFunction):
    def function(self, x):
        # x shape: (N, K) — N muestras, K clases
        # shift por max en cada fila para estabilidad numérica
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x):
        """
        Jacobiana de softmax para cada muestra.

        Para un vector s = softmax(x) de dim K:
          J_ij = s_i * (delta_ij - s_j)

        Retorna shape (N, K, K) — un jacobiano K×K por cada muestra.
        Usar solo cuando se necesite la derivada aislada de softmax.
        En backprop con cross-entropy usar gradiente combinado: (ŷ - y)
        """
        s = self.function(x)  # (N, K)
        # diag(s) por fila - s·sᵀ por fila
        return np.einsum("ij,jk->ijk", s, np.eye(s.shape[-1])) - np.einsum(
            "ij,ik->ijk", s, s
        )

    def gradient_with_cross_entropy(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Gradiente combinado Softmax + Cross-Entropy (lo que realmente se usa en backprop).

        ∂L/∂z = ŷ - y    donde z es la entrada pre-activación (logits)

        Derivación:
          L = -Σ_i y_i · log(ŷ_i)       (cross-entropy)
          ŷ = softmax(z)

          ∂L/∂z_i = ŷ_i - y_i           (resultado elegante del par softmax+CE)

        y_pred: (N, K) probabilidades predichas
        y_true: (N, K) one-hot
        retorna: (N, K)
        """
        return (y_pred - y_true) / y_pred.shape[0]
