from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from nncore import History


class TrainingStrategy(ABC):
    """
    Contrato base para todas las estrategias de entrenamiento.

    Una estrategia implementa UN paso por época y devuelve
    las métricas (loss, accuracy, grad_norm).

    Recibe acceso al modelo para poder llamar:
        model.network, model.cost, model.optimizer
        model._accuracy(), model._compute_grad_norm()
    """

    @abstractmethod
    def step(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Union[tuple[float, float, float], "History"]:
        """
        Ejecuta un paso de entrenamiento (una época completa).

        Retorna:
            (loss, accuracy, grad_norm)
            O un History completo       — estrategias con loop interno (FinalAvgStrategy)
        """
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
