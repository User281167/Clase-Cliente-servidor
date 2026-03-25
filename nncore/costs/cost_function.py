from __future__ import annotations

from abc import ABC

import numpy as np


class CostFunction(ABC):
    """
    Clase base para funciones de costo L(y_true, y_pred).

    Convención de shapes:
      y_true : (N, K)  one-hot  — o (N,) enteros según la subclase
      y_pred : (N, K)  probabilidades o logits según la subclase
      N = batch size, K = número de clases

    derivative() devuelve ∂L/∂y_pred, shape (N, K)
    El promedio sobre N es responsabilidad de cada subclase (documentado).
    """

    def __init__(self):
        super().__init__()

    def function(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.function(y_true, y_pred)

    def _parse_y_true(self, y_true: np.ndarray, K: int) -> np.ndarray:
        """
        Normaliza y_true a one-hot (N, K) independientemente del formato de entrada.

        Acepta:
        - (N,)   enteros en {0,...,K-1}
        - (N,1)  enteros
        - (N,K)  one-hot o probabilidades suaves (soft labels)
        """
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            idx = y_true.flatten().astype(int)
            one_hot = np.zeros((len(idx), K), dtype=float)
            one_hot[np.arange(len(idx)), idx] = 1.0
            return one_hot
        elif y_true.ndim == 2 and y_true.shape[1] == K:
            return y_true  # ya está en el formato correcto
        else:
            raise ValueError(
                f"y_true shape {y_true.shape} incompatible con K={K} clases."
            )
