from .optimizer import Optimizer


# ──────────────────────────────────────────────
# SGD  —  Stochastic Gradient Descent
# ──────────────────────────────────────────────
#
#  La regla de actualización más simple:
#
#    W = W - η · ∂L/∂W
#    b = b - η · ∂L/∂b
#
#  η = learning rate — cuánto nos movemos en dirección del gradiente negativo.
#
#  Intuición geométrica:
#    El gradiente ∂L/∂W apunta hacia la dirección de mayor AUMENTO de L.
#    Restarlo mueve W hacia donde L DECRECE — descenso del gradiente.
#
#  Problemas:
#    - η grande  → pasos grandes → puede diverger u oscilar
#    - η pequeño → converge pero muy lento
#    - En valles elongados oscila en la dirección de mayor curvatura
#      y avanza lento en la dirección de menor curvatura
#
#  A pesar de sus limitaciones, SGD bien tuneado compite con Adam en MNIST.
#
class SGD(Optimizer):
    """
    W ← W - η · ∂L/∂W
    b ← b - η · ∂L/∂b
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, layers: list) -> None:
        for layer in layers:
            if not self._has_params(layer):
                continue
            layer.weights -= self.lr * layer.d_weights
            layer.bias -= self.lr * layer.d_bias
