import numpy as np

from .optimizer import Optimizer


# ──────────────────────────────────────────────
# Adam  —  Adaptive Moment Estimation
# ──────────────────────────────────────────────
#
#  Combina Momentum (primer momento) + RMSProp (segundo momento).
#  El optimizador más usado en la práctica.
#
#  Estado por parámetro:
#    m_W = primer momento  (media de gradientes)       — como Momentum
#    v_W = segundo momento (media de gradientes²)      — como RMSProp
#
#  Actualización:
#
#    m_W = β1·m_W + (1-β1)·∂L/∂W
#    v_W = β2·v_W + (1-β2)·(∂L/∂W)²
#
#  Corrección de bias (crítica en los primeros pasos):
#
#    m̂_W = m_W / (1 - β1^t)
#    v̂_W = v_W / (1 - β2^t)
#
#  Sin esta corrección, en t=1 m_W ≈ 0 y v_W ≈ 0 porque se inicializan
#  en cero — los primeros pasos serían casi nulos.
#  La corrección compensa ese sesgo inicial.
#
#  Actualización final:
#
#    W = W - η · m̂_W / (√v̂_W + ε)
#
#  Hiperparámetros estándar [Kingma & Ba, 2014]:
#    β1 = 0.9    (momentum)
#    β2 = 0.999  (RMSProp)
#    ε  = 1e-8
#    η  = 0.001
#
#  Para MNIST estos defaults funcionan bien sin tunear.
#
class Adam(Optimizer):
    """
    Combina Momentum + RMSProp con corrección de bias.

    m_W ← β1·m_W + (1-β1)·∂L/∂W
    v_W ← β2·v_W + (1-β2)·(∂L/∂W)²
    m̂_W = m_W / (1 - β1^t)
    v̂_W = v_W / (1 - β2^t)
    W   ← W - η · m̂_W / (√v̂_W + ε)

    beta1   : decay primer momento  (default 0.9)
    beta2   : decay segundo momento (default 0.999)
    epsilon : estabilidad numérica  (default 1e-8)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # contador de pasos — para corrección de bias
        self.m = {}  # primer momento   id(layer) -> {"W": m_W, "b": m_b}
        self.v = {}  # segundo momento  id(layer) -> {"W": v_W, "b": v_b}

    def step(self, layers: list) -> None:
        self.t += 1

        for layer in layers:
            if not self._has_params(layer):
                continue

            lid = id(layer)
            if lid not in self.m:
                self.m[lid] = {
                    "W": np.zeros_like(layer.weights),
                    "b": np.zeros_like(layer.bias),
                }
                self.v[lid] = {
                    "W": np.zeros_like(layer.weights),
                    "b": np.zeros_like(layer.bias),
                }

            m, v = self.m[lid], self.v[lid]

            # Actualizar momentos
            m["W"] = self.beta1 * m["W"] + (1 - self.beta1) * layer.d_weights
            m["b"] = self.beta1 * m["b"] + (1 - self.beta1) * layer.d_bias
            v["W"] = self.beta2 * v["W"] + (1 - self.beta2) * layer.d_weights**2
            v["b"] = self.beta2 * v["b"] + (1 - self.beta2) * layer.d_bias**2

            # Corrección de bias
            m_hat_W = m["W"] / (1 - self.beta1**self.t)
            m_hat_b = m["b"] / (1 - self.beta1**self.t)
            v_hat_W = v["W"] / (1 - self.beta2**self.t)
            v_hat_b = v["b"] / (1 - self.beta2**self.t)

            layer.weights -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def reset(self) -> None:
        self.t = 0
        self.m = {}
        self.v = {}
