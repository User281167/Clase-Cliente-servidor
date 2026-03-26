from abc import ABC


class Optimizer(ABC):
    """
    Clase base para optimizadores.

    Un optimizador actualiza los parámetros W y b de cada capa
    usando los gradientes ∂L/∂W y ∂L/∂b calculados en backward.

    El optimizador NO conoce la loss ni calcula gradientes —
    solo recibe gradientes ya calculados y decide cómo aplicarlos.

    step(layers) : realiza una actualización de parámetros
    reset()      : reinicia estado interno (momentum, cache, etc.)
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError(f"learning_rate debe ser > 0. Recibido: {learning_rate}")
        self.lr = learning_rate

    def step(self, layers: list) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def _has_params(self, layer) -> bool:
        """Verifica que la capa tiene parámetros aprendibles (no Dropout, etc.)"""
        return layer.weights is not None and layer.d_weights is not None
