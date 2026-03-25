import numpy as np

from .cost_function import CostFunction


# ──────────────────────────────────────────────
# Cross-Entropy  +  Softmax  (acopladas)
# ──────────────────────────────────────────────
#
#  Separar Softmax de CrossEntropy es tentador pero introduce
#  inestabilidad numérica. El par acoplado es la práctica estándar.
#
#  FORWARD  (y_true one-hot, ŷ = softmax(z))
#
#    L = - (1/N) Σ_n Σ_k  y_nk · log(ŷ_nk)
#
#  Como y_true es one-hot, solo el índice verdadero k* contribuye:
#
#    L = - (1/N) Σ_n  log(ŷ_{n,k*})
#
#  BACKWARD  (gradiente respecto a los logits z, no a ŷ)
#
#    ∂L/∂z_i = (1/N)(ŷ_i - y_i)
#
#  Este resultado elegante viene de aplicar chain rule sobre
#  L(softmax(z)) — la Jacobiana de softmax y la derivada de log
#  se cancelan mutuamente dejando solo (ŷ - y).
#
#  DISEÑO: esta clase recibe logits z (pre-softmax), NO probabilidades.
#  Así el forward y backward son numéricamente estables.
#
class CrossEntropy(CostFunction):
    """
    Cross-Entropy acoplada con Softmax.

    y_true : (N, K) one-hot         ← cambio respecto a versión anterior
    y_pred : (N, K) logits (z)      ← NO probabilidades, mejora estabilidad
    """

    def _softmax(self, z):
        # Método interno — no exponer como softmax independiente.
        # La clase de activación Softmax existe para eso.
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def function(self, y_true, y_pred):
        """
        L = -(1/N) Σ_n Σ_k  y_nk · log(ŷ_nk + ε)

        ε = 1e-12 evita log(0) si alguna probabilidad colapsa a 0.
        """
        K = y_pred.shape[1]
        y_true = self._parse_y_true(y_true, K)
        y_hat = self._softmax(y_pred)
        eps = 1e-12

        return -np.mean(np.sum(y_true * np.log(y_hat + eps), axis=1))

    def derivative(self, y_true, y_pred):
        """
        ∂L/∂z = (1/N)(ŷ - y)

        Gradiente respecto a logits z — listo para backprop directamente.
        """
        K = y_pred.shape[1]
        y_true = self._parse_y_true(y_true, K)
        y_hat = self._softmax(y_pred)

        return (y_hat - y_true) / y_pred.shape[0]
