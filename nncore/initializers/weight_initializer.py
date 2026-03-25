from abc import ABC

import numpy as np


class WeightInitializer(ABC):
    """
    Clase base para inicialización de pesos.

    La inicialización importa porque determina:
      - Si los gradientes explotan o desaparecen en las primeras iteraciones
      - Qué tan rápido converge el entrenamiento
      - Si las activaciones se saturan desde el inicio

    Convención: retorna W de shape (input_size, output_size)
    Los bias se inicializan en cero por separado en la capa.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        return self.initialize(input_size, output_size)
