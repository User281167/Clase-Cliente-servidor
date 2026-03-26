import numpy as np

from .optimizer import Optimizer


# ──────────────────────────────────────────────
# RMSProp  —  Root Mean Square Propagation
# ──────────────────────────────────────────────
#
#  Problema que resuelve: distintos parámetros tienen gradientes
#  de escalas muy distintas. Un η global es demasiado grande
#  para algunos y demasiado pequeño para otros.
#
#  Idea: adaptar η por parámetro según la magnitud reciente
#  de sus gradientes.
#
#    s_W = β·s_W + (1-β)·(∂L/∂W)²       acumula gradientes² (EMA)
#    W   = W - η · ∂L/∂W / (√s_W + ε)
#
#  β = decay rate, típico 0.9 o 0.99
#  ε = 1e-8, evita división por cero
#
#  Interpretación:
#    Si ∂L/∂W ha sido grande recientemente → s_W grande → paso pequeño
#    Si ∂L/∂W ha sido pequeño             → s_W pequeño → paso grande
#
#  El learning rate efectivo por parámetro es:
#
#    η_eff = η / √s_W
#
#  RMSProp fue propuesto por Hinton en un curso de Coursera (2012),
#  nunca publicado formalmente — uno de los optimizadores más citados
#  sin paper oficial.
#
#  Bueno cuando los gradientes varían mucho entre capas
#  Bueno para redes recurrentes
#
class RMSProp(Optimizer):
    """
    s_W ← β·s_W + (1-β)·(∂L/∂W)²
    W   ← W - η · ∂L/∂W / (√s_W + ε)

    beta    : decay rate (default 0.9)
    epsilon : estabilidad numérica (default 1e-8)
    """

    def __init__(
        self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}  # id(layer) -> {"W": s_W, "b": s_b}

    def step(self, layers: list) -> None:
        for layer in layers:
            if not self._has_params(layer):
                continue

            lid = id(layer)
            if lid not in self.cache:
                self.cache[lid] = {
                    "W": np.zeros_like(layer.weights),
                    "b": np.zeros_like(layer.bias),
                }

            s = self.cache[lid]

            # EMA de gradientes al cuadrado
            s["W"] = self.beta * s["W"] + (1 - self.beta) * layer.d_weights**2
            s["b"] = self.beta * s["b"] + (1 - self.beta) * layer.d_bias**2

            layer.weights -= (
                self.lr * layer.d_weights / (np.sqrt(s["W"]) + self.epsilon)
            )
            layer.bias -= self.lr * layer.d_bias / (np.sqrt(s["b"]) + self.epsilon)

    def reset(self) -> None:
        self.cache = {}
