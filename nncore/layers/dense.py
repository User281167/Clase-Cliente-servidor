import numpy as np

from nncore.activations import Softmax

from .layer import Layer


# ──────────────────────────────────────────────
# Dense  —  capa completamente conectada
# ──────────────────────────────────────────────
#
#  FORWARD
#  ───────
#  Dado un batch X de shape (N, n_in):
#
#    Z = X · W + b          shape: (N, n_out)
#    A = f(Z)               shape: (N, n_out)
#
#  donde W shape (n_in, n_out), b shape (1, n_out) — broadcast sobre N
#  f es la función de activación
#
#  BACKWARD
#  ────────
#  Recibe delta = ∂L/∂A de shape (N, n_out)
#  Necesita calcular:
#
#  1) Gradiente respecto a Z (pre-activación):
#
#       δZ = delta ⊙ f'(Z)       shape: (N, n_out)
#
#     ⊙ = producto elemento a elemento (Hadamard)
#
#  2) Gradiente respecto a W:
#
#       ∂L/∂W = Xᵀ · δZ          shape: (n_in, n_out)
#
#  3) Gradiente respecto a b:
#
#       ∂L/∂b = Σ_n δZ_n          shape: (1, n_out)
#             = sum(δZ, axis=0)
#
#  4) Gradiente respecto a X (para propagar a capa anterior):
#
#       ∂L/∂X = δZ · Wᵀ           shape: (N, n_in)
#
#  CASO ESPECIAL — Softmax en capa de salida:
#  Si la activación es Softmax y se usa con CrossEntropy,
#  el delta que llega ya ES (ŷ - y)/N (gradiente combinado).
#  En ese caso f'(Z) NO se aplica aquí — CrossEntropy.derivative()
#  ya lo absorbió. La capa detecta esto automáticamente.
#
class Dense(Layer):
    """
    Capa densa: A = f(X·W + b)

    Parámetros aprendibles: W (n_in, n_out), b (1, n_out)
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Guarda X y Z para usar en backward.

        X : (N, n_in)
        Z : (N, n_out)  — pre-activación, necesaria para f'(Z)
        A : (N, n_out)  — salida post-activación
        """
        self.X = X
        self.Z = np.dot(X, self.weights) + self.bias
        return self.activation(self.Z)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta : ∂L/∂A  shape (N, n_out)

        Si la activación es Softmax, delta ya viene como (ŷ - y)/N
        desde CrossEntropy.derivative() — no aplicar f'(Z) de nuevo.
        """
        if isinstance(self.activation, Softmax):
            # delta = (ŷ - y)/N, ya es ∂L/∂Z directamente
            dZ = delta
        else:
            # Regla general: δZ = delta ⊙ f'(Z)
            dZ = delta * self.activation.derivative(self.Z)

        self.d_weights = np.dot(self.X.T, dZ)  # (n_in, n_out)
        self.d_bias = np.sum(dZ, axis=0, keepdims=True)  # (1, n_out)

        return np.dot(dZ, self.weights.T)  # (N, n_in)

    def __repr__(self):
        return (
            f"Dense(in={self.input_size}, "
            f"out={self.output_size}, "
            f"activation={self.activation.__class__.__name__})"
        )
