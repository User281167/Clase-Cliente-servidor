from __future__ import annotations

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np


class ActivationFunction(ABC):
    """
    Clase base abstracta para funciones de activación.

    Toda función de activación f: R -> R (o R^n -> R^n) debe implementar:
      - function(x)   : la función f(x)
      - derivative(x) : la derivada f'(x) respecto a x

    El método __call__ delega a function(x) para uso natural: relu(x)
    """

    def __init__(self):
        super().__init__()

    def function(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def derivative(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def plot(self, x_range=(-10, 10), num_points=1000):
        """Grafica f(x) y f'(x) lado a lado."""
        x = np.linspace(x_range[0], x_range[1], num_points)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x, self.function(x), color="steelblue")
        axes[0].set_title(f"{self.__class__.__name__}  —  f(x)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("f(x)")
        axes[0].axhline(0, color="k", linewidth=0.8)
        axes[0].axvline(0, color="k", linewidth=0.8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(x, self.derivative(x), color="tomato")
        axes[1].set_title(f"{self.__class__.__name__}  —  f'(x)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("f'(x)")
        axes[1].axhline(0, color="k", linewidth=0.8)
        axes[1].axvline(0, color="k", linewidth=0.8)
        axes[1].grid(alpha=0.3)

        plt.suptitle(self.__class__.__name__, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def __call__(self, x):
        return self.function(x)
