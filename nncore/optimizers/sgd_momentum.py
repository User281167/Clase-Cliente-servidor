import numpy as np

from .optimizer import Optimizer


# ──────────────────────────────────────────────
# SGD + Momentum
# ──────────────────────────────────────────────
#
#  Agrega un término de "velocidad" v que acumula gradientes pasados.
#  Análogo a una bola rodando — acumula inercia en direcciones consistentes
#  y amortigua oscilaciones en direcciones que cambian de signo.
#
#    v_W = β·v_W - η·∂L/∂W
#    W   = W + v_W
#
#  β = coeficiente de momentum, típico 0.9
#    β=0   → equivale a SGD puro
#    β=0.9 → 90% de velocidad anterior + 10% gradiente actual
#    β→1   → demasiada inercia, puede sobrepasar mínimos
#
#  Ventaja sobre SGD:
#    En un valle elongado, los gradientes en la dirección corta
#    se cancelan entre sí (oscilan ±), mientras que en la dirección
#    larga se acumulan → avanza rápido donde debe, suave donde oscila.
#
#  Estado interno: v_W y v_b por capa — se inicializan en cero
#  y se crean la primera vez que step() ve cada capa.
#
class SGDMomentum(Optimizer):
    """
    v_W ← β·v_W - η·∂L/∂W
    W   ← W + v_W

    β : momentum (default 0.9)
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta debe estar en [0, 1). Recibido: {beta}")
        self.beta = beta
        self.velocity = {}  # id(layer) -> {"W": v_W, "b": v_b}

    def step(self, layers: list) -> None:
        for layer in layers:
            if not self._has_params(layer):
                continue

            lid = id(layer)
            if lid not in self.velocity:
                # Primera vez — inicializar velocidad en cero
                self.velocity[lid] = {
                    "W": np.zeros_like(layer.weights),
                    "b": np.zeros_like(layer.bias),
                }

            v = self.velocity[lid]
            v["W"] = self.beta * v["W"] - self.lr * layer.d_weights
            v["b"] = self.beta * v["b"] - self.lr * layer.d_bias

            layer.weights += v["W"]
            layer.bias += v["b"]

    def reset(self) -> None:
        self.velocity = {}
