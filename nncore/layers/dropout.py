import numpy as np

from .layer import Layer


# ──────────────────────────────────────────────
# Dropout
# ──────────────────────────────────────────────
#
#  Técnica de regularización que "apaga" neuronas aleatoriamente
#  durante training para prevenir co-adaptación y overfitting.
#
#  FORWARD — training
#  ──────────────────
#  Sea A la entrada de shape (N, n):
#
#    mask ~ Bernoulli(1-p)     shape: (N, n)   — 1 activa, 0 apagada
#    A_out = A ⊙ mask / (1-p)
#
#  La división por (1-p) es "inverted dropout":
#  mantiene E[A_out] = A sin importar p, así inference
#  no necesita ningún ajuste de escala.
#
#  Demostración:
#    E[mask_i] = (1-p)
#    E[A_out_i] = E[A_i · mask_i / (1-p)]
#               = A_i · E[mask_i] / (1-p)
#               = A_i · (1-p) / (1-p)
#               = A_i  ✓
#
#  FORWARD — inference
#  ───────────────────
#    A_out = A    (pass-through, sin mask, sin escala)
#
#  Gracias a inverted dropout no hay diferencia de escala
#  entre training e inference.
#
#  BACKWARD — solo durante training
#  ─────────────────────────────────
#  La misma mask del forward se reutiliza:
#
#    ∂L/∂A = ∂L/∂A_out ⊙ mask / (1-p)
#
#  Las neuronas apagadas en forward NO reciben gradiente —
#  consistente con que no participaron en el cómputo.
#
#  INTUICIÓN
#  ─────────
#  Entrenar con dropout ≈ entrenar un ensemble de 2^n sub-redes
#  distintas que comparten pesos. En inference se usa la red
#  completa como aproximación del promedio del ensemble.
#
#  Usar en capas ocultas
#  NUNCA en capa de salida — eliminarías clases enteras
#  NUNCA durante evaluación/inferencia
#
#  Valores típicos de p (probabilidad de APAGAR):
#    p = 0.2  → dropout suave, buen punto de partida para MNIST
#    p = 0.5  → dropout agresivo, máxima regularización
#    p > 0.5  → raramente útil, destruye demasiada información
#
class Dropout(Layer):
    """
    Capa de Dropout con inverted dropout.

    p          : probabilidad de APAGAR una neurona (default 0.5)
    training   : True durante entrenamiento, False en inference

    No tiene pesos aprendibles — W y b son None.
    No aplica activación — actúa sobre la salida de la capa anterior.
    """

    def __init__(self, p: float = 0.5):
        """
        p : tasa de dropout — fracción de neuronas a apagar.
            p=0.0 equivale a no hacer nada.
            p=1.0 apaga todo (inútil).
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p debe estar en [0, 1). Recibido: {p}")

        # Dropout no tiene input/output fijo ni activación ni inicializador
        # Llamamos ABS directamente para no forzar esos parámetros
        super().__init__()

        self.p = p
        self.training = True  # se cambia desde Network antes de forward
        self.mask = None  # se guarda en forward para reusar en backward

        # Sin parámetros aprendibles
        self.weights = None
        self.bias = None
        self.d_weights = None
        self.d_bias = None

        # Dimensiones desconocidas hasta el primer forward
        self.input_size = None
        self.output_size = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Training  : A_out = X ⊙ mask / (1-p)
        Inference : A_out = X
        """
        if not self.training or self.p == 0.0:
            return X

        # Bernoulli(1-p): 1 con prob (1-p), 0 con prob p
        self.mask = (np.random.rand(*X.shape) >= self.p).astype(float)
        return X * self.mask / (1.0 - self.p)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Propaga gradiente solo por las neuronas que estuvieron activas.

        ∂L/∂X = delta ⊙ mask / (1-p)
        """
        if not self.training or self.p == 0.0:
            return delta

        if self.mask is None:
            raise RuntimeError("backward() llamado antes de forward()")

        return delta * self.mask / (1.0 - self.p)

    def __repr__(self):
        return f"Dropout(p={self.p}, training={self.training})"
