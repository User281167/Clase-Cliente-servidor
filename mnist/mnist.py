# %% [markdown]
# ## Inhabilitar uso de threads en numpy
#
# Inhabilitar cuando se use procesos físicos para evitar competencia de recursos
#
# Ejecutar antes de import numpy
#
# **Si se ejecuto con threads y se quiere ejecutar con cpu física primero se debe reiniciar el kernel y ejecutar la celda, igualmente si se ejecuto la celda y quiere activar de nuevo los threads para numpy**

from __future__ import annotations

import argparse
import os
import platform
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple

import cpuinfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

# %%
# import os
# # ── Configuración de threads ──────────────────────────────
# os.environ["OMP_NUM_THREADS"]     = "1"
# os.environ["MKL_NUM_THREADS"]     = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ## Librerías - utilidades
# %%
from kagglehub import dataset_download
from loky import get_reusable_executor


# %%
def print_system_info():
    print("=== SYSTEM INFO ===")
    print("OS:", platform.system(), platform.release())
    print("Machine:", platform.machine())

    info = cpuinfo.get_cpu_info()
    print("CPU:", info["brand_raw"])
    print("Architecture:", info["arch"])
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Logical threads:", psutil.cpu_count(logical=True))

    ram = psutil.virtual_memory()
    print("RAM (GB):", round(ram.total / (1024**3), 2))
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"cpu_count: {os.cpu_count()}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM Total (GB): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
        )
        print(f"VRAM Allocated (GB): {torch.cuda.memory_allocated(0) / 1024**3:.2f}")
        print(f"VRAM Reserved (GB): {torch.cuda.memory_reserved(0) / 1024**3:.2f}")

    print("_" * 40)


def time_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed < 1:
            formatted = f"{elapsed * 1000:.2f} ms"
        elif elapsed < 60:
            formatted = f"{elapsed:.2f} s"
        elif elapsed < 3600:
            mins, secs = divmod(elapsed, 60)
            formatted = f"{int(mins)}m {secs:.2f}s"
        else:
            hours, remainder = divmod(elapsed, 3600)
            mins, secs = divmod(remainder, 60)
            formatted = f"{int(hours)}h {int(mins)}m {secs:.2f}s"

        print(f"{func.__name__} took {formatted}")
        return result

    return wrapper


# %% [markdown]
# # Mini Lib

# %% [markdown]
# ## Carga de datos


# %%
class MnistData:
    def __init__(self):
        self.dataset_path = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def download_data(self) -> None:
        self.dataset_path = dataset_download("oddrationale/mnist-in-csv")

    def load_data(self, one_hot: bool = True) -> None:
        train_df = pd.read_csv(self.dataset_path + "/mnist_train.csv")
        test_df = pd.read_csv(self.dataset_path + "/mnist_test.csv")

        self.X_train = train_df.iloc[
            :, 1:
        ].values  # 60000 imágenes de 784 píxeles (28x28)
        self.y_train = train_df.iloc[:, 0].values  # 60000 etiquetas (0-9)
        self.X_test = test_df.iloc[:, 1:].values
        self.y_test = test_df.iloc[:, 0].values

        self.X_train = self.X_train.astype("float32") / 255.0
        self.X_test = self.X_test.astype("float32") / 255.0

        if one_hot:
            self.y_train = self.one_hot_encode(self.y_train)
            self.y_test = self.one_hot_encode(self.y_test)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def plot_random_samples(self, n_samples=10) -> None:
        indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        samples = self.X_train[indices]
        labels = self.y_train[indices]

        # labels a entero si es one-hot
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)

        plt.figure(figsize=(10, 1))

        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap="gray")
            plt.title(f"Label: {labels[i]}")
            plt.axis("off")

        plt.show()

    def train_val_split(self, val_ratio: float = 0.1, seed: int | None = None):
        """
        Particiona X_train/y_train en train+val.
        val_ratio: fracción para validación, ej. 0.1 → 6000 muestras
        """
        np.random.seed(seed)
        n = len(self.X_train)
        np.random.seed()
        idx = np.random.permutation(n)
        split = int(n * val_ratio)
        val_idx, train_idx = idx[:split], idx[split:]

        return (
            self.X_train[train_idx],
            self.y_train[train_idx],
            self.X_train[val_idx],
            self.y_train[val_idx],
        )

    def one_hot_encode(self, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
        """
        Convierte etiquetas enteras a vectores one-hot.

        y: array (N,) con valores en {0,...,9}
        retorna: (N, 10) donde cada fila es e_k
        """
        one_hot = np.zeros((len(y), n_classes), dtype="float32")
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot

    def get_batches(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
    ):
        """
        Generador de mini-batches.

        Para N muestras y tamaño B, produce ceil(N/B) batches.
        Si shuffle=True, permuta antes de particionar (esencial en SGD/mini-batch
        para no introducir sesgo por orden de clase).
        """
        n = len(X)
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            yield X[batch_idx], y[batch_idx]


# %% [markdown]
# ## Funciones de activación


# %%
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


# ──────────────────────────────────────────────
# ReLU  —  Rectified Linear Unit
# ──────────────────────────────────────────────
#
#          ⎧  0   si x ≤ 0
#  f(x) =  ⎨
#          ⎩  x   si x > 0
#
#  = max(0, x)
#
#          ⎧  0   si x < 0
#  f'(x) = ⎨  ?   si x = 0   (subgradiente, usamos 0)
#          ⎩  1   si x > 0
#
#  Ventaja : gradiente constante → no hay vanishing gradient en zona activa
#  Problema: "dying ReLU" — neuronas que quedan siempre en x<0 dejan de aprender
#
class ReLU(ActivationFunction):
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(x.dtype)


# ──────────────────────────────────────────────
# Softmax
# ──────────────────────────────────────────────
#
#           e^(x_i - max(x))
#  f(x)_i = ─────────────────      i = 0,...,K-1
#            Σ_j e^(x_j - max(x))
#
#  El shift por max(x) es estabilidad numérica (evita overflow), no cambia el resultado:
#    e^(x_i - c) / Σ e^(x_j - c) = e^x_i / Σ e^x_j
#
#  Salida: distribución de probabilidad — Σ_i f(x)_i = 1, f(x)_i ∈ (0,1)
#
#  Jacobiana (derivada completa es una matriz, no un vector):
#
#    ∂f_i/∂x_j = f_i · (δ_ij - f_j)
#
#    En forma matricial para un vector s = softmax(x):
#      J = diag(s) - s·sᵀ     (matriz K×K)
#
#  ⚠️  ADVERTENCIA IMPORTANTE:
#  En la práctica, Softmax se usa SIEMPRE con Cross-Entropy como loss.
#  El gradiente combinado ∂L/∂x = ŷ - y es muy simple (ver CostFunction).
#  La Jacobiana completa RARAMENTE se necesita sola.
#
#  El método derivative() aquí devuelve el jacobiano por fila (útil para
#  visualización o casos edge), PERO en el backward pass de la red
#  usar el gradiente combinado Softmax + CrossEntropy directamente.
#
class Softmax(ActivationFunction):
    def __init__(self, is_output=True):
        super().__init__()
        self.is_output = is_output

    def function(self, x):
        # x shape: (N, K) — N muestras, K clases
        # shift por max en cada fila para estabilidad numérica
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x):
        """
        Jacobiana de softmax para cada muestra.

        Para un vector s = softmax(x) de dim K:
          J_ij = s_i * (delta_ij - s_j)

        Retorna shape (N, K, K) — un jacobiano K×K por cada muestra.
        Usar solo cuando se necesite la derivada aislada de softmax.
        En backprop con cross-entropy usar gradiente combinado: (ŷ - y)

        Lo anterior es la derivada real
        Sin embargo si es capa de salida simplemente se usa el delta de la función de costo
        """

        if self.is_output:
            return np.ones_like(x)  # el gradiente real ya viene en delta desde cost

        s = self.function(x)  # (N, K)
        # diag(s) por fila - s·sᵀ por fila
        return np.einsum("ij,jk->ijk", s, np.eye(s.shape[-1])) - np.einsum(
            "ij,ik->ijk", s, s
        )


# %% [markdown]
# ## Funciones de costo

# %%


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


# ──────────────────────────────────────────────
# Mean Squared Error
# ──────────────────────────────────────────────
#
#  L = (1/N) Σ_n ||y_n - ŷ_n||²
#    = (1/N) Σ_n Σ_k (y_nk - ŷ_nk)²
#
#  ∂L/∂ŷ = (2/N)(ŷ - y)
#
#  Para clasificación con one-hot, MSE penaliza todas las clases por igual
#  lo que no es ideal — CrossEntropy es mejor para clasificación.
#  MSE sí tiene sentido en la capa de salida de un autoencoder o regresión.
#
class MeanSquaredError(CostFunction):
    """
    y_true : (N, K) one-hot o valores continuos
    y_pred : (N, K) salida de la red (post-activación)
    """

    def function(self, y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    def derivative(self, y_true, y_pred):
        """
        ∂L/∂ŷ = (2/N)(ŷ - y)

        Factor 2 se suele absorber en el learning rate,
        algunos frameworks usan (1/N)(ŷ - y) omitiendo el 2.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]


# %% [markdown]
# ## Inicializadores

# %%


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


# ──────────────────────────────────────────────
# Random Normal
# ──────────────────────────────────────────────
#
#  W ~ N(0, 1)
#
#  El problema: sin escalar, la varianza de la salida crece con input_size.
#  Si input_size=784 (MNIST), las activaciones explotan inmediatamente.
#
#  ❌ No usar en producción — sirve para demostrar por qué importa
#     la inicialización correcta (experimento didáctico).
#
class RandomNormal(WeightInitializer):
    """
    W ~ N(0, 1)  — sin escalar.
    Solo útil para ilustrar el problema de inicialización naive.
    """

    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std

    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.randn(input_size, output_size).astype(np.float32) * self.std


# %% [markdown]
# ## Capas

# %%


class Layer(ABC):
    """
    Clase base para capas de una red neuronal.

    Una capa transforma una entrada X de shape (N, n_in)
    a una salida de shape (N, n_out) donde N = batch size.

    Todo Layer debe implementar:
      forward(X)      : paso hacia adelante  X -> ŷ
      backward(delta) : paso hacia atrás     ∂L/∂X <- ∂L/∂ŷ

    El backward recibe el gradiente de la loss respecto a la
    SALIDA de esta capa, y devuelve el gradiente respecto a
    la ENTRADA (para propagarlo a la capa anterior).
    Los gradientes de W y b se almacenan en self.d_weights, self.d_bias.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction,
        weight_initializer: WeightInitializer,
    ):
        # weight_initializer debe ser una INSTANCIA, no la clase
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.initializer = weight_initializer
        self.weights = weight_initializer.initialize(input_size, output_size)
        self.bias = np.zeros((1, output_size)).astype(np.float32)

        # Gradientes — se llenan en backward, se leen en el optimizer
        self.d_weights = None
        self.d_bias = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_norm(self):
        return np.linalg.norm(self.d_weights)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in={self.input_size}, "
            f"out={self.output_size}, "
            f"activation={self.activation.__class__.__name__})"
        )


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
#  ⚠️  CASO ESPECIAL — Softmax en capa de salida:
#  Si la activación es Softmax y se usa con CrossEntropy,
#  el delta que llega ya ES (ŷ - y)/N (gradiente combinado).
#  En ese caso f'(Z) NO se aplica aquí — CrossEntropy.derivative()
#  ya lo absorbió. La Función de activación gestiona esto
#
class Dense(Layer):
    """
    Capa densa: A = f(X·W + b)

    Parámetros aprendibles: W (n_in, n_out), b (1, n_out)
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        X : (N, n_in)
        Z : (N, n_out)  pre-activación
        A : (N, n_out)  post-activación
        """
        self.X = X

        # Z = XW + b
        self.Z = X @ self.weights
        self.Z += self.bias

        # activación
        return self.activation(self.Z)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta : ∂L/∂A  shape (N, n_out)

        Retorna:
            ∂L/∂X
        """

        # δZ = δA ⊙ f'(Z)
        dZ = delta * self.activation.derivative(self.Z)

        # gradientes parámetros
        self.d_weights = self.X.T @ dZ
        self.d_bias = np.sum(dZ, axis=0, keepdims=True)

        # gradiente para capa anterior
        return dZ @ self.weights.T

    def __repr__(self):
        return (
            f"Dense(in={self.input_size}, "
            f"out={self.output_size}, "
            f"activation={self.activation.__class__.__name__})"
        )


# %% [markdown]
# ## Optimizadores


# %%
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


# ──────────────────────────────────────────────
# SGD  —  Stochastic Gradient Descent
# ──────────────────────────────────────────────
#
#  La regla de actualización más simple:
#
#    W = W - η · ∂L/∂W
#    b = b - η · ∂L/∂b
#
#  η = learning rate — cuánto nos movemos en dirección del gradiente negativo.
#
#  Intuición geométrica:
#    El gradiente ∂L/∂W apunta hacia la dirección de mayor AUMENTO de L.
#    Restarlo mueve W hacia donde L DECRECE — descenso del gradiente.
#
#  Problemas:
#    - η grande  → pasos grandes → puede diverger u oscilar
#    - η pequeño → converge pero muy lento
#    - En valles elongados oscila en la dirección de mayor curvatura
#      y avanza lento en la dirección de menor curvatura
#
#  A pesar de sus limitaciones, SGD bien tuneado compite con Adam en MNIST.
#
class SGD(Optimizer):
    """
    W ← W - η · ∂L/∂W
    b ← b - η · ∂L/∂b
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, layers: list) -> None:
        for layer in layers:
            if not self._has_params(layer):
                continue
            layer.weights -= self.lr * layer.d_weights
            layer.bias -= self.lr * layer.d_bias


# %% [markdown]
# ## Red

# %%


# ──────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────
#
#  Representa el grafo de cómputo de la red neuronal.
#  Conecta capas secuencialmente y gestiona el flujo
#  de datos en ambas direcciones.
#
#  FORWARD PASS
#  ────────────
#  Dada entrada X de shape (N, n_in), propaga por cada capa:
#
#    A_0 = X
#    A_l = f_l(A_{l-1})    l = 1, ..., L
#
#  La salida A_L es la predicción de la red.
#
#  BACKWARD PASS
#  ─────────────
#  Recibe el gradiente inicial δ = ∂L/∂A_L de la loss,
#  propaga en orden inverso:
#
#    δ_{l-1} = layer_l.backward(δ_l)    l = L, ..., 1
#
#  Cada capa acumula d_weights y d_bias internamente.
#  El optimizer los lee después con optimizer.step(layers).
#
#  TRAINING vs INFERENCE
#  ──────────────────────
#  Dropout se comporta distinto en cada modo.
#  Network gestiona el flag training en todas las capas
#  antes de cada forward — el usuario no tiene que hacerlo.
#
class Network:
    """
    Grafo de cómputo secuencial.

    No conoce la loss, el optimizer ni los datos —
    solo sabe propagar hacia adelante y hacia atrás.

    Uso básico:
        net = Network()
        net.add(Dense(784, 128, Tanh(), XavierUniform()))
        net.add(Dropout(0.3))
        net.add(Dense(128, 10, Softmax(), XavierUniform()))

        y_pred = net.forward(X)          # inference por default
        y_pred = net.forward(X, training=True)  # training
        grad   = net.backward(delta)
    """

    def __init__(self):
        super().__init__()
        self.layers = []

    # ──────────────────────────────────────────
    # Construcción
    # ──────────────────────────────────────────

    def add(self, layer) -> "Network":
        """
        Agrega una capa al final de la red.
        Retorna self para permitir encadenamiento:

            net.add(Dense(...)).add(Dropout(...)).add(Dense(...))
        """
        self.layers.append(layer)
        return self

    def __len__(self) -> int:
        return len(self.layers)

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Propaga X por todas las capas en orden.

        training : True  durante entrenamiento (activa Dropout, etc.)
                   False durante inference (default)

        A_0 = X
        A_l = layer_l(A_{l-1})    l = 1,...,L

        Retorna A_L de shape (N, n_out)
        """
        self._set_training(training)

        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def __call__(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(X, training)

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Propaga el gradiente en orden inverso.

        delta : ∂L/∂A_L — gradiente de la loss respecto
                a la salida de la última capa. Shape (N, n_out).

        Cada capa acumula d_weights y d_bias.
        Retorna el gradiente respecto a la entrada X (raramente
        necesario fuera de redes más complejas).

        ⚠️  Llamar siempre DESPUÉS de forward(training=True).
            backward sin forward previo usa cachés desactualizados.
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def reset_weights(self) -> None:
        """
        Reinicializa W y b de todas las capas entrenables.
        Llamar entre experimentos para comparación justa.
        """
        for layer in self.trainable_layers():
            layer.weights = layer.initializer.initialize(
                layer.input_size, layer.output_size
            )
            layer.bias = np.zeros((1, layer.output_size))

    # ──────────────────────────────────────────
    # Parámetros
    # ──────────────────────────────────────────

    def trainable_layers(self) -> list:
        """
        Retorna solo las capas con parámetros aprendibles (W, b).
        El optimizer llama esto para saber qué actualizar.
        """
        return [l for l in self.layers if l.weights is not None]

    def parameter_count(self) -> int:
        """
        Cuenta el total de parámetros aprendibles de la red.

        Para una capa Dense(n_in, n_out):
          params = n_in * n_out  (W)  +  n_out  (b)
        """
        total = 0
        for layer in self.trainable_layers():
            total += layer.weights.size + layer.bias.size
        return total

    # ──────────────────────────────────────────
    # Persistencia  —  guardar y cargar pesos
    # ──────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        """
        Guarda W y b de cada capa entrenable en un archivo .npz.

        Formato: weights_0, bias_0, weights_1, bias_1, ...
        El índice corresponde a la posición en trainable_layers().
        """
        data = {}
        for i, layer in enumerate(self.trainable_layers()):
            data[f"weights_{i}"] = layer.weights
            data[f"bias_{i}"] = layer.bias
        np.savez(path, **data)

    def load_weights(self, path: str) -> None:
        """
        Carga pesos desde un .npz guardado con save_weights().

        ⚠️  La arquitectura debe ser idéntica a la que generó el archivo.
        """
        data = np.load(path)
        layers = self.trainable_layers()

        for i, layer in enumerate(layers):
            key_w = f"weights_{i}"
            key_b = f"bias_{i}"

            if key_w not in data or key_b not in data:
                raise ValueError(
                    f"Archivo no contiene pesos para capa {i}. "
                    f"¿La arquitectura coincide con el archivo?"
                )

            if data[key_w].shape != layer.weights.shape:
                raise ValueError(
                    f"Shape mismatch en capa {i}: "
                    f"archivo={data[key_w].shape}, "
                    f"red={layer.weights.shape}"
                )

            layer.weights = data[key_w]
            layer.bias = data[key_b]

    # ──────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────

    def _set_training(self, training: bool) -> None:
        """
        Propaga el flag training a todas las capas que lo usen (Dropout).
        Se llama automáticamente en forward().
        """
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training

    def summary(self) -> None:
        """
        Imprime la arquitectura de la red con shapes y parámetros.

        Ejemplo:
        ┌─────────────────────────────────────────────────┐
        │                  Network Summary                │
        ├──────────────────────┬───────────┬──────────────┤
        │ Layer                │ Shape     │ Params       │
        ├──────────────────────┼───────────┼──────────────┤
        │ Dense (Tanh)         │ 784→128   │ 100,480      │
        │ Dropout (p=0.3)      │    —      │       0      │
        │ Dense (Softmax)      │ 128→10    │   1,290      │
        ├──────────────────────┼───────────┼──────────────┤
        │ Total params         │           │ 101,770      │
        └─────────────────────────────────────────────────┘
        """
        sep = "─" * 52
        total = 0

        print(f"┌{sep}┐")
        print(f"│{'Network Summary':^52}│")
        print(f"├{'─' * 22}┬{'─' * 11}┬{'─' * 16}┤")
        print(f"│{'Layer':<22}│{'Shape':^11}│{'Params':>14}  │")
        print(f"├{'─' * 22}┼{'─' * 11}┼{'─' * 16}┤")

        for layer in self.layers:
            if layer.weights is not None:
                name = f"{layer.__class__.__name__} ({layer.activation.__class__.__name__})"
                shape = f"{layer.input_size}→{layer.output_size}"
                params = layer.weights.size + layer.bias.size
                total += params
                print(f"│{name:<22}│{shape:^11}│{params:>14,}  │")
            else:
                name = repr(layer)
                print(f"│{name:<22}│{'—':^11}│{'0':>14}  │")

        print(f"├{'─' * 22}┴{'─' * 11}┴{'─' * 16}┤")
        print(f"│{'Total params':<22}  {total:>26,}  │")
        print(f"└{sep}┘")

    def __repr__(self) -> str:
        lines = [f"Network({len(self.layers)} layers)"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {repr(layer)}")
        return "\n".join(lines)


# %% [markdown]
# ## Historial


# %%
class History:
    """
    Registra métricas por epoch durante el entrenamiento.

    Atributos:
        train_loss     : list[float]  — loss promedio por epoch (train)
        train_accuracy : list[float]  — accuracy por epoch (train)
        train_grad_norm : norma de Frobenius promedio del gradiente por epoch
                          ||∂L/∂W||_F = sqrt(Σ_ij (∂L/∂W_ij)²)
                          Útil para detectar:
                            - grad_norm → 0   vanishing gradient
                            - grad_norm → ∞   exploding gradient
                            - grad_norm estable → entrenamiento saludable
        val_loss       : list[float]  — loss promedio por epoch (val)
        val_accuracy   : list[float]  — accuracy por epoch (val)
        epochs         : list[int]    — índices de epoch registrados
    """

    def __init__(self, output_dir: str = None):
        self.train_loss = []
        self.train_accuracy = []
        self.train_grad_norm = []
        self.val_loss = []
        self.val_accuracy = []
        self.epochs = []
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def record(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        grad_norm: float = None,
        val_loss: float = None,
        val_acc: float = None,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_acc)
        self.train_grad_norm.append(grad_norm if grad_norm is not None else 0.0)

        if val_loss is not None:
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_acc)

    def plot(self, show_grad_norm: bool = True) -> None:
        """
        Grafica las métricas registradas durante el entrenamiento.

        show_grad_norm=True agrega un tercer panel con la norma del gradiente.
        """
        has_val = len(self.val_loss) > 0
        has_grad = show_grad_norm and any(g > 0 for g in self.train_grad_norm)
        n_panels = 3 if has_grad else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

        # ── Loss ──
        axes[0].plot(self.epochs, self.train_loss, label="Train", color="steelblue")
        if has_val:
            axes[0].plot(
                self.epochs, self.val_loss, label="Val", color="tomato", linestyle="--"
            )
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # ── Accuracy ──
        axes[1].plot(self.epochs, self.train_accuracy, label="Train", color="steelblue")
        if has_val:
            axes[1].plot(
                self.epochs,
                self.val_accuracy,
                label="Val",
                color="tomato",
                linestyle="--",
            )
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # ── Grad norm ──
        if has_grad:
            axes[2].plot(
                self.epochs, self.train_grad_norm, color="seagreen", label="||∂L/∂W||_F"
            )
            axes[2].set_title("Gradient Norm")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("||∂L/∂W||_F")
            axes[2].legend()
            axes[2].grid(alpha=0.3)

        plt.suptitle("Training History", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "training_plot.png")
            plt.savefig(path, dpi=120)

        plt.show()

    def plot_comparison(
        self,
        other: "History",
        label_self: str = "Model A",
        label_other: str = "Model B",
    ) -> None:
        """
        Compara dos historiales — útil para comparar modos de GD,
        optimizadores, o arquitecturas distintas.

        Ejemplo:
            history_adam.plot_comparison(history_sgd,
                                         label_self="Adam",
                                         label_other="SGD")
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        pairs = [
            (self.train_loss, other.train_loss, "Loss", "Loss"),
            (self.train_accuracy, other.train_accuracy, "Accuracy", "Accuracy"),
            (
                self.train_grad_norm,
                other.train_grad_norm,
                "Gradient Norm",
                "||∂L/∂W||_F",
            ),
        ]
        colors = [
            ("steelblue", "tomato"),
            ("steelblue", "tomato"),
            ("seagreen", "darkorange"),
        ]

        for ax, (a, b, title, ylabel), (ca, cb) in zip(axes, pairs, colors):
            ax.plot(self.epochs, a, label=label_self, color=ca)
            ax.plot(other.epochs, b, label=label_other, color=cb, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def summary_table(self) -> None:
        """
        Imprime tabla resumen con best/last por métrica.

        ┌─────────────────────┬────────────┬────────────┐
        │ Métrica             │    Best    │    Last    │
        ├─────────────────────┼────────────┼────────────┤
        │ train_loss          │   0.0812   │   0.0934   │
        ...
        """
        metrics = [
            ("train_loss", self.train_loss, min),
            ("train_accuracy", self.train_accuracy, max),
            ("train_grad_norm", self.train_grad_norm, min),
        ]

        if self.val_loss:
            metrics += [
                ("val_loss", self.val_loss, min),
                ("val_accuracy", self.val_accuracy, max),
            ]

        sep = "─" * 45
        print(f"┌{sep}┐")
        print(f"│{'Training Summary':^45}│")
        print(f"├{'─' * 21}┬{'─' * 11}┬{'─' * 11}┤")
        print(f"│{'Métrica':<21}│{'Best':^11}│{'Last':^11}│")
        print(f"├{'─' * 21}┼{'─' * 11}┼{'─' * 11}┤")

        for name, values, best_fn in metrics:
            if values:
                best = best_fn(values)
                last = values[-1]
                print(f"│{name:<21}│{best:^11.4f}│{last:^11.4f}│")

        print(f"└{'─' * 21}┴{'─' * 11}┴{'─' * 11}┘")

    def save_csv(self, filename: str = "history.csv") -> None:
        data = {
            "epoch": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_accuracy,
            "grad_norm": self.train_grad_norm,
        }

        if self.val_loss:
            data["val_loss"] = self.val_loss
            data["val_acc"] = self.val_accuracy

        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)

    def save_plots(self):
        # Loss
        plt.figure()
        plt.plot(self.epochs, self.train_loss, label="Train")
        if self.val_loss:
            plt.plot(self.epochs, self.val_loss, label="Val")

        plt.legend()
        plt.title("Loss")
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close()

        # Accuracy
        plt.figure()
        plt.plot(self.epochs, self.train_accuracy, label="Train")
        if self.val_accuracy:
            plt.plot(self.epochs, self.val_accuracy, label="Val")

        plt.legend()
        plt.title("Accuracy")
        plt.savefig(os.path.join(self.output_dir, "accuracy.png"))
        plt.close()

        # Grad norm
        if any(g > 0 for g in self.train_grad_norm):
            plt.figure()
            plt.plot(self.epochs, self.train_grad_norm)
            plt.title("Gradient Norm")
            plt.savefig(os.path.join(self.output_dir, "grad_norm.png"))
            plt.close()

    def save_summary(self):
        path = os.path.join(self.output_dir, "summary.txt")

        with open(path, "w") as f:
            for name, values, fn in [
                ("train_loss", self.train_loss, min),
                ("train_acc", self.train_accuracy, max),
                ("grad_norm", self.train_grad_norm, min),
            ]:
                if values:
                    f.write(f"{name}: best={fn(values):.4f}, last={values[-1]:.4f}\n")

            if self.val_loss:
                f.write(
                    f"val_loss: best={min(self.val_loss):.4f}, last={self.val_loss[-1]:.4f}\n"
                )
                f.write(
                    f"val_acc: best={max(self.val_accuracy):.4f}, last={self.val_accuracy[-1]:.4f}\n"
                )

    def save_all(self):
        self.save_csv()
        self.save_plots()
        self.save_summary()

    def __repr__(self) -> str:
        if not self.epochs:
            return "History(vacío)"

        s = (
            f"History(epochs={self.epochs[-1]}, "
            f"train_loss={self.train_loss[-1]:.4f}, "
            f"train_acc={self.train_accuracy[-1]:.4f}, "
            f"grad_norm={self.train_grad_norm[-1]:.4f}"
        )

        if self.val_loss:
            s += (
                f", val_loss={self.val_loss[-1]:.4f}, "
                f"val_acc={self.val_accuracy[-1]:.4f}"
            )
        return s + ")"


# %% [markdown]
# ## Modelo

# %%
from concurrent.futures import as_completed
from typing import Union


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


class FullBatchStrategy(TrainingStrategy):
    """
    Batch GD clásico — un paso con TODOS los datos.

    W = W - η · (1/N) Σ_n ∂L_n/∂W


    Ventaja  : gradiente exacto sobre el dataset completo
    Desventaja: lento y costoso en memoria para datasets grandes,
                puede quedar atrapado en mínimos locales amplios
    """

    def step(self, model, X, y):
        y_pred = model.network.forward(X, training=True)
        loss = model.cost.function(y, y_pred)
        delta = model.cost.derivative(y, y_pred)
        model.network.backward(delta)
        model.optimizer.step(model.network.trainable_layers())

        return (
            float(loss),
            float(model._accuracy(y, y_pred)),
            model._compute_grad_norm(),
        )


class WeightAvgStrategy(TrainingStrategy):
    """
    Mini-batch Averaging — cada batch parte del MISMO W_0,
    obtiene su W_b, y al final se promedia: W = avg(W_1,...,W_M).

    La próxima época parte del promedio, no de W_M.

    Algoritmo por época:
    época 1:
        W_0 = pesos actuales (fijo para todos los batches)

        batch 1 → forward+backward+step → W_1  (desde W_0)
        batch 2 → forward+backward+step → W_2  (desde W_0)
        batch 3 → forward+backward+step → W_3  (desde W_0)
        ...

        W_new = (W_1 + W_2 + ... + W_M) / M   ← promedio de destinos
        W = W_new                              ← arranca la siguiente época

    época 2:
        batch 1 → forward+backward+step → W_1  (desde W)
        batch 2 → forward+backward+step → W_2  (desde W)
        batch 3 → forward+backward+step → W_3  (desde W)
        W = (W_1 + W_2 + W_3) / M
    """

    def __init__(self, batch_size: int, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def step(self, model, X, y):
        n = len(X)

        if self.shuffle:
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

        layers = model.network.trainable_layers()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]
        W_acc = [np.zeros_like(l.weights) for l in layers]
        b_acc = [np.zeros_like(l.bias) for l in layers]

        epoch_loss = epoch_acc = epoch_gnorm = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            # Cada batch parte del mismo punto W_0
            for i, layer in enumerate(layers):
                layer.weights = W0[i].copy()
                layer.bias = b0[i].copy()

            X_b = X[start : start + self.batch_size]
            y_b = y[start : start + self.batch_size]
            y_pred = model.network.forward(X_b, training=True)
            loss = model.cost.function(y_b, y_pred)
            delta = model.cost.derivative(y_b, y_pred)
            model.network.backward(delta)

            # Aplicar paso del optimizer para obtener W_b
            model.optimizer.step(layers)

            # Acumular W_b — el destino de este batch
            for i, layer in enumerate(layers):
                W_acc[i] += layer.weights
                b_acc[i] += layer.bias

            epoch_loss += float(loss)
            epoch_acc += float(model._accuracy(y_b, y_pred))
            epoch_gnorm += model._compute_grad_norm()
            n_batches += 1

        # Cargar promedio de destinos
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / n_batches
            layer.bias = b_acc[i] / n_batches

        return epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm / n_batches


class FinalAvgStrategy(TrainingStrategy):
    """
    Cada worker entrena local_epochs independientemente.
    El History refleja el progreso REAL del worker 0 época por época.
    Promedia pesos UNA sola vez al final.

    A diferencia de WeightAvgStrategy (que registra 1 punto por época global),
    esta estrategia expone el entrenamiento interno completo en el History.
    """

    def __init__(
        self,
        n_workers: int,
        local_epochs: int = 200,
        batch_size: int = None,
        shuffle=True,
    ):
        self.n_workers = n_workers
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def step(self, model, X, y) -> "History":
        # para el historial inyectar los valores de tests
        X_val = getattr(self, "_X_val", None)
        y_val = getattr(self, "_y_val", None)

        layers = model.network.trainable_layers()
        n = len(X)
        chunk_size = n // self.n_workers
        history = History()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]
        W_acc = [np.zeros_like(l.weights) for l in layers]
        b_acc = [np.zeros_like(l.bias) for l in layers]

        if self.shuffle:
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

        for k in range(self.n_workers):
            # Cada worker parte de W_0
            for i, layer in enumerate(layers):
                layer.weights = W0[i].copy()
                layer.bias = b0[i].copy()

            start = k * chunk_size
            end = (k + 1) * chunk_size if k < self.n_workers - 1 else n
            X_k, y_k = X[start:end], y[start:end]

            for ep in range(self.local_epochs):
                if self.batch_size is not None:
                    idx_k = np.random.permutation(len(X_k))
                    ep_gnorm = 0.0
                    n_b = 0

                    for b in range(0, len(X_k), self.batch_size):
                        X_b = X_k[idx_k[b : b + self.batch_size]]
                        y_b = y_k[idx_k[b : b + self.batch_size]]

                        y_pred = model.network.forward(X_b, training=True)
                        delta = model.cost.derivative(y_b, y_pred)
                        model.network.backward(delta)
                        model.optimizer.step(layers)

                        ep_gnorm += model._compute_grad_norm()
                        n_b += 1

                    ep_gnorm /= n_b
                else:
                    y_pred = model.network.forward(X_k, training=True)
                    delta = model.cost.derivative(y_k, y_pred)
                    model.network.backward(delta)

                    model.optimizer.step(layers)
                    ep_gnorm = model._compute_grad_norm()

                # Solo worker 0 alimenta el History
                if k == 0:
                    y_ep = model.network.forward(X_k, training=False)
                    ep_loss = float(model.cost.function(y_k, y_ep))
                    ep_acc = float(model._accuracy(y_k, y_ep))

                    val_loss = val_acc = None
                    if X_val is not None:
                        val_loss, val_acc = model._evaluate_split(X_val, y_val)

                    history.record(ep + 1, ep_loss, ep_acc, ep_gnorm, val_loss, val_acc)

            for i, layer in enumerate(layers):
                W_acc[i] += layer.weights
                b_acc[i] += layer.bias

        # Promedio único al final
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / self.n_workers
            layer.bias = b_acc[i] / self.n_workers

        return history  # ← History completo


def _process_weight_avg_step(W0, b0, X_b, y_b, model_fn):
    """
    W0:         Pesos iniciales del batch
    b0:         Bias iniciales del batch
    X_b:        Datos de entrenamiento del batch
    y_b:        Datos de predicción del batch
    model_fun:  Construcción del modelo local para el batch no tocar el modelo (W, b) padre
                ej:
                    def build_model():
                        return Model(
                            get_network(),
                            MeanSquaredError(),
                            SGD(learning_rate=0.05),
                            strategy=None
                        )
    """
    # Crear modelo nuevo dentro del proceso
    try:
        model = model_fn()
        layers = model.network.trainable_layers()

        # Cargar W0
        for i, layer in enumerate(layers):
            layer.weights = W0[i].copy()
            layer.bias = b0[i].copy()

        # Forward / backward
        y_pred = model.network.forward(X_b, training=True)
        loss = model.cost.function(y_b, y_pred)
        delta = model.cost.derivative(y_b, y_pred)
        model.network.backward(delta)
        model.optimizer.step(layers)

        acc = model._accuracy(y_b, y_pred)
        gnorm = model._compute_grad_norm()

        # Devolver pesos finales
        Wb = [l.weights for l in layers]
        bb = [l.bias for l in layers]

        return (
            Wb,
            bb,
            float(loss),
            float(model._accuracy(y_b, y_pred)),
            float(model._compute_grad_norm()),
        )
    except Exception as e:
        return ("error", str(e))  # no bloquear el proceso padre


class ParallelWeightAvgStrategy(TrainingStrategy):
    """
    Mini-batch Averaging — cada batch parte del MISMO W_0,
    obtiene su W_b, y al final se promedia: W = avg(W_1,...,W_M).

    La próxima época parte del promedio, no de W_M.

    Algoritmo por época:
    época 1:
        W_0 = pesos actuales (fijo para todos los batches)

        batch 1 → forward+backward+step → W_1  (desde W_0, en procesos diferente)
        batch 2 → forward+backward+step → W_2  (desde W_0, en procesos diferente)
        batch 3 → forward+backward+step → W_3  (desde W_0, en procesos diferente)
        ...

        W_new = (W_1 + W_2 + ... + W_M) / M   ← promedio de destinos
        W = W_new                              ← arranca la siguiente época

    época 2:
        batch 1 → forward+backward+step → W_1  (desde W, en procesos diferente)
        batch 2 → forward+backward+step → W_2  (desde W, en procesos diferente)
        batch 3 → forward+backward+step → W_3  (desde W, en procesos diferente)
        W = (W_1 + W_2 + W_3) / M
    """

    def __init__(
        self, batch_size, model_fn, n_workers=None, reserved_cores=2, shuffle=True
    ):
        """
        model_fun:  Construcción del modelo local para el batch no tocar el modelo padre
                ej:
                    def build_model():
                        return Model(
                            get_network(),
                            MeanSquaredError(),
                            SGD(learning_rate=0.05),
                            strategy=None
                        )
        n_workers: cores físicos a usar obligatorio sino
        reserved_cores: cores a dejar para no bloquear procesos del SO

        El comportamiento con 8 batches y `max_procs=4`:
            activos: [b0, b1, b2, b3]  → terminan
            activos: [b4, b5, b6, b7]  → terminan
            promedia y retorna

        Inhabilitar threads de numpy para evitar competencia por recursos
        """

        self.batch_size = batch_size
        self.model_fn = model_fn
        self.shuffle = shuffle

        physical = psutil.cpu_count(logical=False) or 1

        if n_workers is None:
            self.n_workers = max(1, physical - reserved_cores)
        else:
            self.n_workers = n_workers

        # Prioridad solo en Windows
        if os.name == "nt":
            try:
                psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            except psutil.AccessDenied:
                print("Error HIGH_PRIORITY_CLASS")
                pass  # sin permisos de admin, ignorar

    def step(self, model, X, y):
        """
        El proceso padre:
            - Calcula W₀
            - Divide dataset en M particiones
            - Lanza M procesos
            - Cada proceso recibe:
                - W₀
                - Su partición de datos
            - Recoge W_b de cada proceso
            - Promedia
        """
        n = len(X)

        if self.shuffle:
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

        layers = model.network.trainable_layers()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]

        starts = list(range(0, n, self.batch_size))
        n_batches = len(starts)

        batches = [
            (
                W0,
                b0,
                X[s : s + self.batch_size],
                y[s : s + self.batch_size],
                self.model_fn,
            )
            for s in starts
        ]

        max_procs = min(n_batches, self.n_workers)
        max_procs = max(1, max_procs)
        results = []

        executor = get_reusable_executor(max_workers=max_procs)
        futures = [executor.submit(_process_weight_avg_step, *b) for b in batches]
        results = [
            f.result() for f in as_completed(futures)
        ]  # espera todos antes de continuar

        # Acumuladores
        W_acc = [np.zeros_like(w) for w in W0]
        b_acc = [np.zeros_like(b) for b in b0]

        epoch_loss = epoch_acc = epoch_gnorm = 0.0

        for result in results:
            if isinstance(result[0], str) and result[0] == "error":
                # raise RuntimeError(f"Proceso hijo falló: {result[1]}")
                print(f"Proceso hijo falló: {result[1]}")
                n_batches -= n_batches - 1  # reducir para el promedio final
                continue

            Wb, bb, loss, acc, gnorm = result

            for i in range(len(W_acc)):
                W_acc[i] += Wb[i]
                b_acc[i] += bb[i]

            epoch_loss += loss
            epoch_acc += acc
            epoch_gnorm += gnorm

        if n_batches == 0:
            print(f"Todos los procesos fallaron en esta época")
            return 0, 0, 0, 0

        # Promedio final
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / n_batches
            layer.bias = b_acc[i] / n_batches

        return (epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm / n_batches)


class GradientAvgStrategy(TrainingStrategy):
    """
    Promedia los gradientes de todas las muestras en cada época.

    Algoritmo por época:
    época 1:
        W_0 = pesos actuales (fijo para todos los batches)

        batch 1 → forward+backward → W_grad_1  (desde W_0)
        batch 2 → forward+backward → W_grad_2  (desde W_0)
        batch 3 → forward+backward → W_grad_3  (desde W_0)
        ...

        W_grad = (W_1 + W_2 + ... + W_M) / M   ← promedio de destinos
        W = SGD(W_grad)                        ← arranca la siguiente época

    época 2:
        batch 1 → forward+backward → W_grad_1  (desde W)
        batch 2 → forward+backward → W_grad_2  (desde W)
        batch 3 → forward+backward → W_grad_3  (desde W)
        W_grad = (W_1 + W_2 + W_3) / M
        W = SGD(W_grad)
    """

    def __init__(self, batch_size: int, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._local_epoch = 0

        self.metrics = pd.DataFrame(
            columns=[
                "local_epoch",
                "time",
                "loss",
                "gnorm",
                "throughput",
                "batch_size",
            ]
        )

    def step(self, model, X, y):
        t0 = time.perf_counter()
        self._local_epoch += 1

        n = len(X)

        if self.shuffle:
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

        layers = model.network.trainable_layers()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]
        W_acc = [np.zeros_like(l.weights) for l in layers]
        b_acc = [np.zeros_like(l.bias) for l in layers]

        epoch_loss = epoch_acc = epoch_gnorm = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            # Cada batch parte del mismo punto W_0
            for i, layer in enumerate(layers):
                layer.weights = W0[i].copy()
                layer.bias = b0[i].copy()

            X_b = X[start : start + self.batch_size]
            y_b = y[start : start + self.batch_size]
            y_pred = model.network.forward(X_b, training=True)
            loss = model.cost.function(y_b, y_pred)
            delta = model.cost.derivative(y_b, y_pred)
            model.network.backward(delta)

            # Acumular W_b — el destino de este batch
            for i, layer in enumerate(layers):
                W_acc[i] += layer.d_weights
                b_acc[i] += layer.d_bias

            epoch_loss += float(loss)
            epoch_acc += float(model._accuracy(y_b, y_pred))
            n_batches += 1

        # Cargar promedio de destinos
        for i, layer in enumerate(layers):
            layer.d_weights = W_acc[i] / n_batches
            layer.d_bias = b_acc[i] / n_batches

        # Aplicar paso del optimizer para obtener W_b
        model.optimizer.step(layers)
        epoch_gnorm += model._compute_grad_norm()

        elapsed = time.perf_counter() - t0
        throughput = self.batch_size / elapsed

        self.metrics.loc[len(self.metrics)] = {
            "local_epoch": self._local_epoch,
            "time": elapsed,
            "loss": epoch_loss / n_batches,
            "gnorm": epoch_gnorm,
            "throughput": throughput,
            "batch_size": self.batch_size,
        }

        return epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm


# %%
class Model:
    """
    Orquesta el ciclo de entrenamiento delegando el paso
    por época a una TrainingStrategy intercambiable.

    Uso:
        model = Model(network, CrossEntropy(), Adam(), MiniBatchStrategy(64))
        history = model.fit(X_train, y_train, epochs=20)
    """

    def __init__(
        self,
        network: "Network",
        cost: "CostFunction",
        optimizer: "Optimizer",
        strategy: "TrainingStrategy" = None,
    ):
        self.network = network
        self.cost = cost
        self.optimizer = optimizer
        self.strategy = strategy or FullBatchStrategy()

    # ──────────────────────────────────────────
    # Entrenamiento
    # ──────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True,
        verbose_epoch: int = 1,
    ) -> "History":
        if isinstance(self.strategy, FinalAvgStrategy):
            self.strategy._X_val = X_val
            self.strategy._y_val = y_val

        history = History()

        for epoch in range(1, epochs + 1):
            result = self.strategy.step(self, X_train, y_train)

            # FinalAvgStrategy devuelve History directamente
            if isinstance(result, History):
                return result  # ya tiene todo registrado

            loss, acc, gnorm = result

            val_loss = val_acc = None
            if X_val is not None:
                val_loss, val_acc = self._evaluate_split(X_val, y_val)

            history.record(epoch, loss, acc, gnorm, val_loss, val_acc)

            if verbose and (
                epoch == 1 or epoch % verbose_epoch == 0 or epoch == epochs
            ):
                self._print_epoch(epoch, epochs, loss, acc, gnorm, val_loss, val_acc)

        return history

    def _compute_grad_norm(self) -> float:
        """
        Norma de Frobenius de los gradientes de TODOS los pesos.

        ||∂L/∂W||_F = sqrt( Σ_layers Σ_ij (∂L/∂W_ij)² )

        Un valor único por paso que resume la magnitud global
        del gradiente en la red completa.

        Interpretación:
        < 0.001  → posible vanishing gradient
        > 100    → posible exploding gradient
        estable  → entrenamiento bien condicionado
        """
        total = 0.0

        for layer in self.network.trainable_layers():
            if layer.d_weights is not None:
                total += np.sum(layer.d_weights**2)
                total += np.sum(layer.d_bias**2)

        return float(np.sqrt(total))

    # ──────────────────────────────────────────
    # Evaluación y predicción
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción en modo inference (Dropout desactivado).
        Retorna probabilidades shape (N, K).
        """
        return self.network.forward(X, training=False)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna la clase predicha por argmax.
        shape (N,) con valores en {0,...,K-1}
        """
        return np.argmax(self.predict(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """
        Evalúa la red sobre un conjunto completo.
        Retorna dict con loss y accuracy.
        """
        loss, acc = self._evaluate_split(X, y)
        if verbose:
            print(f"Loss: {loss:.4f}  |  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

        return {"loss": loss, "accuracy": acc}

    def _evaluate_split(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.network.forward(X, training=False)
        loss = float(self.cost.function(y, y_pred))
        acc = float(self._accuracy(y, y_pred))

        return loss, acc

    def confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Matriz de confusión para clasificación multiclase.

        C[i, j] = número de muestras de clase i predichas como clase j

        Diagonal principal : predicciones correctas
        Fuera de diagonal  : errores — C[i,j] dice "clase i confundida con j"

        Para MNIST (10 clases) produce matriz 10×10.
        """
        y_pred = self.predict_classes(X)
        y_true = np.argmax(y, axis=1) if y.ndim == 2 else y.flatten().astype(int)
        K = len(np.unique(y_true))
        cm = np.zeros((K, K), dtype=int)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        if plot:
            self._plot_confusion_matrix(cm)

        return cm

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Visualiza la matriz de confusión con anotaciones.
        Normaliza por fila para mostrar % de acierto por clase.
        """
        K = cm.shape[0]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Conteos absolutos ──
        im0 = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Confusion Matrix — Conteos")
        plt.colorbar(im0, ax=axes[0])

        for i in range(K):
            for j in range(K):
                axes[0].text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )

        axes[0].set_xlabel("Predicho")
        axes[0].set_ylabel("Real")
        axes[0].set_xticks(range(K))
        axes[0].set_yticks(range(K))

        # ── Normalizada por fila (% accuracy por clase) ──
        im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Confusion Matrix — Normalizada")
        plt.colorbar(im1, ax=axes[1])

        for i in range(K):
            for j in range(K):
                axes[1].text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        axes[1].set_xlabel("Predicho")
        axes[1].set_ylabel("Real")
        axes[1].set_xticks(range(K))
        axes[1].set_yticks(range(K))

        plt.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Reporte por clase: precision, recall, F1-score.

        Precision_k = TP_k / (TP_k + FP_k)
        De todo lo que predije como k, ¿cuánto era realmente k?

        Recall_k = TP_k / (TP_k + FN_k)
        De todo lo que era k, ¿cuánto predije correctamente?

        F1_k = 2 · (Precision_k · Recall_k) / (Precision_k + Recall_k)
        Media armónica — balance entre precision y recall

        Útil para detectar qué dígitos confunde más la red.
        """
        cm = self.confusion_matrix(X, y, plot=False)
        K = cm.shape[0]
        eps = 1e-8

        print(f"\n{'─' * 52}")
        print(
            f"{'Clase':^8} {'Precision':^12} {'Recall':^12} {'F1':^12} {'Samples':^8}"
        )
        print(f"{'─' * 52}")

        precisions = recalls = f1s = 0.0

        for k in range(K):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            samples = cm[k, :].sum()

            precisions += precision
            recalls += recall
            f1s += f1

            print(f"{k:^8} {precision:^12.4f} {recall:^12.4f} {f1:^12.4f} {samples:^8}")

        print(f"{'─' * 52}")
        print(
            f"{'avg':^8} {precisions / K:^12.4f} {recalls / K:^12.4f} {f1s / K:^12.4f}"
        )
        print(f"{'─' * 52}\n")

    # ──────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Accuracy = (1/N) Σ_n 1[argmax(ŷ_n) == argmax(y_n)]

        Funciona con y_true one-hot o enteros.
        """
        pred = np.argmax(y_pred, axis=1)

        if y_true.ndim == 2:
            true = np.argmax(y_true, axis=1)
        else:
            true = y_true.flatten().astype(int)

        return np.mean(pred == true)

    # ──────────────────────────────────────────
    # Verbose
    # ──────────────────────────────────────────

    def _print_epoch(
        self,
        epoch: int,
        total: int,
        loss: float,
        acc: float,
        gnorm: float = None,
        val_loss=None,
        val_acc=None,
    ) -> None:
        bar_len = 20
        filled = int(bar_len * epoch / total)

        bar = "█" * filled + "░" * (bar_len - filled)
        line = (
            f"Epoch {epoch:>4}/{total} [{bar}] "
            f"loss={loss:.4f}  acc={acc:.4f}  "
            f"||g||={gnorm:.4f}"
        )

        if val_loss is not None:
            line += f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

        print(line)

    def __repr__(self) -> str:
        return (
            f"Model(\n"
            f"  network={repr(self.network)},\n"
            f"  cost={self.cost.__class__.__name__},\n"
            f"  optimizer={self.optimizer.__class__.__name__}\n"
            f")"
        )


# %% [markdown]
# # Preparación

# %%
# Datos
data = MnistData()
data.download_data()


# %%
data.load_data(one_hot=True)
X_train, y_train, X_test, y_test = data.get_data()

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# %%
SEED = 42


def get_network():
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))
    return net


net = get_network()


@time_wrapper
def run_experiment(strategy, epochs=200, verbose_epoch=50):
    print_system_info()

    np.random.seed(SEED)
    net.reset_weights()
    np.random.seed()

    model = Model(net, MeanSquaredError(), SGD(learning_rate=0.05), strategy=strategy)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
        verbose_epoch=verbose_epoch,
    )

    history.plot()
    model.evaluate(X_test, y_test)
    model.confusion_matrix(X_test, y_test)
    model.classification_report(X_test, y_test)

    return history


# %% [markdown]
# ## Reporte estadístico para repeticiones

# %%
from dataclasses import dataclass, field

# ──────────────────────────────────────────────
# Resultado de una repetición
# ──────────────────────────────────────────────


@dataclass
class TrialResult:
    seed: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


# ──────────────────────────────────────────────
# Resultado agregado del estudio
# ──────────────────────────────────────────────


@dataclass
class StudyResult:
    strategy_name: str
    trials: list[TrialResult] = field(default_factory=list)

    # ── Estadísticas ──────────────────────────

    def _stats(self, values: list[float]) -> dict:
        a = np.array(values)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    @property
    def train_acc_stats(self):
        return self._stats([t.train_acc for t in self.trials])

    @property
    def val_acc_stats(self):
        return self._stats([t.val_acc for t in self.trials])

    @property
    def train_loss_stats(self):
        return self._stats([t.train_loss for t in self.trials])

    @property
    def val_loss_stats(self):
        return self._stats([t.val_loss for t in self.trials])

    # ── Print ─────────────────────────────────

    def _print_stats(self, label: str, stats: dict) -> None:
        print(f"\n--- {self.strategy_name} | {label} ---")
        print(f"  Media:          {stats['mean']:.4f}")
        print(f"  Desv. Estándar: {stats['std']:.4f}")
        print(f"  Mínimo:         {stats['min']:.4f}")
        print(f"  Máximo:         {stats['max']:.4f}")
        print(f"  Mediana:        {stats['median']:.4f}")

    def report(self) -> None:
        self._print_stats("Accuracy  TEST", self.val_acc_stats)
        self._print_stats("Accuracy  TRAIN", self.train_acc_stats)
        self._print_stats("Loss      TEST", self.val_loss_stats)
        self._print_stats("Loss      TRAIN", self.train_loss_stats)


# ──────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────
@time_wrapper
def run_study(
    strategy_factory,  # callable que devuelve una estrategia nueva: lambda: WeightAvgStrategy(6000)
    network_factory,  # callable que devuelve una red nueva:         get_network
    cost,
    optimizer_factory,  # callable que devuelve un optimizer nuevo:    lambda: SGD(0.05)
    X_train,
    y_train,
    X_val,
    y_val,
    repetitions: int = 30,
    epochs: int = 200,
    seeds: list[int] | int = None,
    verbose: bool = True,
) -> StudyResult:
    """
    Repite el entrenamiento `repetitions` veces con seeds distintas.
    Cada repetición usa una instancia nueva de red, optimizer y estrategia.
    No modifica ninguna clase existente.

    Parámetros:
        strategy_factory  : lambda sin args → TrainingStrategy
        network_factory   : lambda sin args → Network
        optimizer_factory : lambda sin args → Optimizer
        seeds             : lista de ints; si None se generan automáticamente; si int se generan automáticamente reproducibilidad

    Retorna:
        StudyResult con estadísticas agregadas
    """
    if seeds is None or isinstance(seeds, int):
        rng = np.random.default_rng(seeds)
        seeds = rng.integers(0, 100_000, size=repetitions).tolist()

    strategy_name = strategy_factory().__class__.__name__
    result = StudyResult(strategy_name=strategy_name)

    for i, seed in enumerate(seeds):
        if verbose:
            print(
                f"[{strategy_name}] rep {i + 1:>3}/{repetitions}  seed={seed}", end="\r"
            )

        # ── Inicialización reproducible ──
        np.random.seed(seed)
        net = network_factory()
        np.random.seed()  # liberar seed para training aleatorio

        strategy = strategy_factory()
        optimizer = optimizer_factory()
        model = Model(net, cost, optimizer, strategy=strategy)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            X_val=X_val,
            y_val=y_val,
            verbose=False,  # silenciar epochs individuales
        )

        # ── Métricas finales de la repetición ──
        train_loss = history.train_loss[-1]
        train_acc = history.train_accuracy[-1]
        val_loss = history.val_loss[-1]
        val_acc = history.val_accuracy[-1]

        result.trials.append(
            TrialResult(
                seed=seed,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )

    if verbose:
        print()  # salto de línea tras el \r

    return result


# %% [markdown]
# # Train


# %% [markdown]
# # Servidor de parámetros

# %% [markdown]
# ## Secuencia

# %% [markdown]
# ![Secuenciad entrenamiento distribuido](./server_sequence.png)

# %% [markdown]
# ## Server

# %%
"""
═══════════════════════
Entrenamiento distribuido sobre TCP con la misma interfaz que las estrategias locales.

Arquitectura
────────────
                    ┌────────────────────────┐
                    │  ServerTrainingStrategy│  (ABC)  maneja TCP, serialización,
                    │                        │         broadcast, aggregation loop
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │ ServerWeightAvgStrategy│  implementa step() — lógica Batch-Avg
                    └────────────────────────┘

                    ┌──────────────────────┐
                    │    ClientStrategy    │  (ABC)  maneja conexión, loop de mensajes
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │DistributedBatchAvg   │  implementa run()  — ejecuta forward/back
                    │Strategy              │
                    └──────────────────────┘

Protocolo de mensajes (JSON + pickle sobre TCP)
───────────────────────────────────────────────
  SERVER → CLIENT
    {"type": "assign",   "worker_id": int, "seed": int, "start": int, "n_batches": int}
    {"type": "weights",  "payload": <pickle bytes b64>}   ← W0, b0
    {"type": "step"}                                      ← ejecuta un paso
    {"type": "done"}                                      ← cierra

  CLIENT → SERVER
    {"type": "ready",    "worker_id": int}
    {"type": "result",   "worker_id": int, "payload": <pickle bytes b64>}
                          payload = (Wb, bb, loss, acc, gnorm)

Uso — Servidor
──────────────
    strategy = ServerWeightAvgStrategy(batch_size=256, min_workers=2)
    strategy.start_server(port=9999)

    model = Model(net, MeanSquaredError(), SGD(0.05), strategy=strategy)
    history = model.fit(X_train, y_train, epochs=50)

Uso — Cliente
─────────────
    model = Model(net, MeanSquaredError(), SGD(0.05))   # optimizer ignorado
    strategy = DistributedWeightAvgStrategy("192.168.1.10", 9999)
    strategy.connect()
    strategy.run(model, X_train, y_train)               # blocking loop
"""

import base64
import json
import logging
import pickle
import select
import selectors
import socket
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

log = logging.getLogger("distributed_training")
log.setLevel(logging.DEBUG)

if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_h)

log.propagate = False

# ══════════════════════════════════════════════════════════════════════════════
# Helpers de red — framing simple: 4 bytes (big-endian) de longitud + payload
# ══════════════════════════════════════════════════════════════════════════════


def _send_msg(sock, obj):
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(len(raw).to_bytes(4, "big") + raw)


def _send_safe(wid, sock, msg):
    try:
        _send_msg(sock, msg)
        return None
    except Exception as e:
        log.warning(f"Worker {wid} error in send: {e}")
        return wid


def _recv_msg(sock):
    header = _recv_exact(sock, 4)
    length = int.from_bytes(header, "big")
    raw = _recv_exact(sock, length)
    return pickle.loads(raw)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()

    while len(buf) < n:
        chunk = sock.recv(n - len(buf))

        if not chunk:
            raise ConnectionError("Socket cerrado prematuramente")

        buf.extend(chunk)

    return bytes(buf)


# ══════════════════════════════════════════════════════════════════════════════
# ABC — Servidor
# ══════════════════════════════════════════════════════════════════════════════


class ServerTrainingStrategy(ABC):
    """
    Base para estrategias de entrenamiento distribuido en el lado servidor.

    Gestiona:
      - TCP accept loop en hilo separado
      - Registro y reconexión de workers
      - Broadcast de pesos iniciales
      - Recolección de resultados
      - Serialización/deserialización (JSON + pickle b64)

    Subclases deben implementar `step()` con la lógica de agregación.
    """

    # ── Configuración ──────────────────────────────────────────────────────

    CONNECT_TIMEOUT = 120  # segundos esperando min_workers al inicio
    WORKER_TIMEOUT = 60  # segundos esperando resultado de un worker
    RECONNECT_WINDOW = 30  # segundos que un worker tiene para reconectarse
    PING_INTERVAL = 30  # segundos entre health-pings (heartbeat futuro)

    def __init__(self, min_workers: int = 1):
        self.min_workers = min_workers

        self._server_sock: Optional[socket.socket] = None
        self._port: Optional[int] = None

        # worker_id → socket activo
        self._workers: dict[int, socket.socket] = {}
        self._workers_lock = threading.Lock()

        # worker_id → metadatos asignados (seed, start, n_batches)
        self._assignments: dict[int, dict] = {}

        self._accept_thread: Optional[threading.Thread] = None
        self._running = False
        self._next_worker_id = 0

        # Se dispara cuando len(_workers) >= min_workers
        self._ready_event = threading.Event()

        # para broadcast
        self._pool = ThreadPoolExecutor(max_workers=32)

    # ── Ciclo de vida del servidor ─────────────────────────────────────────

    def start_server(self, port: int = 9999, host: str = "0.0.0.0") -> None:
        """Arranca el servidor TCP y espera min_workers conexiones."""
        self._port = port
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )  # Reuse address
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._server_sock.bind((host, port))
        self._server_sock.listen(32)
        self._running = True

        log.info(
            f"Servidor escuchando en {host}:{port} (min_workers={self.min_workers})"
        )

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # Espera min_workers workers listos — sin polling, Event se dispara
        # desde _handshake_worker en cuanto se alcanza el umbral
        log.info(f"Esperando {self.min_workers} worker(s)...")

        if not self._ready_event.wait(timeout=self.CONNECT_TIMEOUT):
            with self._workers_lock:
                n = len(self._workers)

            raise TimeoutError(
                f"Solo {n}/{self.min_workers} workers conectaron en {self.CONNECT_TIMEOUT}s"
            )

        with self._workers_lock:
            n = len(self._workers)

        log.info(f"{n} worker(s) conectados — servidor listo")

    def stop_server(self) -> None:
        """
        Cierre ordenado:
        1. Envía done a todos los workers y cierra sus sockets
        2. Detiene el accept loop (running=False)
        3. Cierra el server socket
        4. Apaga el thread pool
        El accept thread es daemon — no necesita join(), muere con el proceso
        o cuando el server socket se cierra y OSError interrumpe el accept().
        """
        if not self._running:
            return

        self._running = False

        # 1 — notificar y cerrar workers
        with self._workers_lock:
            for wid, sock in list(self._workers.items()):
                try:
                    _send_msg(sock, {"type": "done"})
                except Exception:
                    pass
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except Exception:
                    pass
            self._workers.clear()

        # 2 & 3 — cerrar server socket interrumpe accept() en el hilo daemon
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

            self._server_sock = None

        # 4 — pool
        self._pool.shutdown(wait=False)
        self._ready_event.clear()
        log.info("Servidor detenido")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop_server()

    # ── Accept loop (hilo daemon) ──────────────────────────────────────────

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)

        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handshake_worker,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handshake_worker(self, conn: socket.socket, addr) -> None:
        """Asigna worker_id y espera 'ready'."""
        try:
            msg = _recv_msg(conn)

            if msg.get("type") != "ready":
                conn.close()
                return

            # ¿reconexión?
            existing_id = msg.get("worker_id")

            with self._workers_lock:
                if existing_id is not None and existing_id in self._assignments:
                    wid = existing_id
                    log.info(f"Worker {wid} reconectado desde {addr}")
                else:
                    wid = self._next_worker_id
                    self._next_worker_id += 1
                    log.info(f"Nuevo worker {wid} desde {addr}")

                self._workers[wid] = conn

                # Disparar event si ya tenemos suficientes workers
                if len(self._workers) >= self.min_workers:
                    self._ready_event.set()

            # Confirmar asignación
            assign = self._assignments.get(wid, {})
            _send_msg(conn, {"type": "assign", "worker_id": wid, **assign})

        except Exception as e:
            log.warning(f"Error en handshake con {addr}: {e}")
            conn.close()

    def _wait_workers(self) -> int | None:
        with self._workers_lock:
            n = len(self._workers)

        if n == 0:
            log.warning("No hay workers activos, esperando reconexion...")

            if not self._ready_event.wait(timeout=self.CONNECT_TIMEOUT):
                log.warning("Timeout esperando workers — saltando epoca")
                return None

            with self._workers_lock:
                n = len(self._workers)

        return n

    # ── Comunicación ───────────────────────────────────────────────────────
    def _broadcast_fast(self, msg):
        # No usa pool
        with self._workers_lock:
            workers = list(self._workers.items())

        for wid, sock in workers:
            dead = _send_safe(wid, sock, msg)
            if dead:
                self._remove_dead([dead])

    def _broadcast_pool(self, msg):
        with self._workers_lock:
            workers = list(self._workers.items())  # [(wid, sock), ...]

        if not workers:
            return

        results = list(
            self._pool.map(lambda ws: _send_safe(ws[0], ws[1], msg), workers)
        )
        dead = [wid for wid in results if wid is not None]

        if dead:
            self._remove_dead(dead)

    def _broadcast_weights(self, W0: list, b0: list) -> None:
        """Envía W0, b0 a todos los workers registrados."""
        msg = {"type": "weights", "payload": (W0, b0)}

        self._broadcast_pool(msg)

    def _broadcast_step(self) -> None:
        """Indica a todos los workers que ejecuten su paso."""
        self._broadcast_fast({"type": "step"})

    def _collect_results(self):
        sel = selectors.DefaultSelector()

        with self._workers_lock:
            items = list(self._workers.items())

        results = []
        dead = []

        for wid, sock in items:
            sel.register(sock, selectors.EVENT_READ, wid)

        deadline = time.time() + self.WORKER_TIMEOUT

        while sel.get_map():
            timeout = max(0, deadline - time.time())
            events = sel.select(timeout)

            if not events:
                break

            for key, _ in events:
                sock = key.fileobj
                wid = key.data

                sel.unregister(sock)

                try:
                    msg = _recv_msg(sock)

                    if msg["type"] != "result":
                        raise ValueError(msg["type"])

                    results.append(msg["payload"])
                except Exception as e:
                    log.warning(f"Worker {wid} falló: {e}")
                    dead.append(wid)

        for key in sel.get_map().values():
            dead.append(key.data)

        self._remove_dead(dead)

        return results

    def _remove_dead(self, wids: list) -> None:
        with self._workers_lock:
            for wid in wids:
                sock = self._workers.pop(wid, None)

                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

                log.warning(f"Worker {wid} removido")

            if not self._workers:
                log.warning("No quedan workers activos")
                self._ready_event.clear()

    # ── Abstracto ─────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, model, X: np.ndarray, y: np.ndarray):
        """Ejecuta una época completa y devuelve (loss, acc, gnorm)."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Estrategia concreta — Servidor: Batch Averaging
# ══════════════════════════════════════════════════════════════════════════════


class ServerWeightAvgStrategy(ServerTrainingStrategy):
    """
    Equivalente distribuido de ParallelWeightAvgStrategy.

    Algoritmo por época:
        1. Captura W₀ de la red local
        2. Asigna a cada worker su slice de datos (seed + start)
        3. Broadcast W₀
        4. Broadcast 'step'  → workers hacen forward+backward+optimizer
        5. Recoge (Wb, bb, loss, acc, gnorm) de cada worker
        6. W_new = promedio(W₁, …, W_M)
        7. Carga W_new en la red local

    Sin solapamiento de datos
    ─────────────────────────
    El servidor genera un seed global y asigna `start` distinto a cada worker.
    Cada worker hace:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        X_b = X[idx[start * batch_size : (start+1) * batch_size]]

    Así todos usan el mismo shuffle pero porciones distintas.
    """

    def __init__(self, batch_size: int = 256, min_workers: int = 1):
        super().__init__(min_workers=min_workers)
        self.batch_size = batch_size
        self._epoch_seed = 0  # incrementa cada época para variar el shuffle

    def _assign_slices(self, n_workers: int) -> int:
        """
        Genera seed y asigna start a cada worker.
        Retorna el seed usado.
        """
        seed = int(time.time()) ^ (self._epoch_seed * 0x9E3779B9)
        self._epoch_seed += 1

        with self._workers_lock:
            wids = list(self._workers.keys())

        for i, wid in enumerate(wids):
            msg = {
                "type": "assign",
                "worker_id": wid,
                "seed": seed,
                "start": i,
                "n_batches": n_workers,
                "batch_size": self.batch_size,
            }

            self._assignments[wid] = msg

            try:
                _send_msg(self._workers[wid], msg)
            except Exception as e:
                log.warning(f"No se pudo asignar slice a worker {wid}: {e}")

        return seed

    def step(self, model, X: np.ndarray, y: np.ndarray):
        """
        Ejecuta una época distribuida y actualiza los pesos del modelo.
        Devuelve (loss, acc, gnorm) — igual que cualquier TrainingStrategy.
        """
        n_workers = self._wait_workers()

        if n_workers is None:
            log.warning("Ningún worker conectado, saltando época")
            return 0, 0, 0

        layers = model.network.trainable_layers()
        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]

        # 1. Asignar slices
        self._assign_slices(n_workers)

        # 2. Broadcast pesos
        self._broadcast_weights(W0, b0)

        # 3. Broadcast step
        self._broadcast_step()

        # 4. Recoger resultados
        results = self._collect_results()

        if not results:
            log.warning("Ningún worker devolvió resultado, saltando época")
            return 0, 0, 0

        # 5. Promediar
        W_acc = [np.zeros_like(w) for w in W0]
        b_acc = [np.zeros_like(b) for b in b0]
        total_loss = total_acc = total_gnorm = 0.0

        for Wb, bb, loss, acc, gnorm in results:
            for i in range(len(W_acc)):
                W_acc[i] += Wb[i]
                b_acc[i] += bb[i]

            total_loss += loss
            total_acc += acc
            total_gnorm += gnorm

        m = len(results)
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / m
            layer.bias = b_acc[i] / m

        return total_loss / m, total_acc / m, total_gnorm / m


from datetime import datetime


class ServerGradientAvgStrategy(ServerTrainingStrategy):
    """
    Similar a WeightAvgStrategy pero promedia gradientes en vez de pesos
        El cliente calcula el gradiente
        El servidor optimiza la media de los gradientes

        batch 1 → calcula ∂L_1/∂W
        batch 2 → calcula ∂L_2/∂W
        grad_avg = avg(∂L_1/∂W, ∂L_2/∂W)
        W_new = W - optimizer(grad_avg)   ← optimizer se aplica UNA vez sobre el promediado
    """

    def __init__(self, batch_size: int = 256, min_workers: int = 1):
        super().__init__(min_workers=min_workers)
        self.batch_size = batch_size
        self._epoch_seed = 0  # incrementa cada época para variar el shuffle

        self.metrics = pd.DataFrame(
            columns=[
                "epoch",  # Época
                "time",  # Tiempo total de la época
                "total_loss",  # Pérdida total
                "accuracy",  # Accuracy global
                "gnorm",  # Norma de gradiente global
                "throughput",  # Throughput (opcional)
                "n_workers",  # Número de workers
            ]
        )

    def _broadcast_weights(self, W0, b0):
        # seed va embebido — cliente no necesita mensaje assign separado
        seed = int(time.time()) ^ (self._epoch_seed * 0x9E3779B9)
        self._epoch_seed += 1
        start = 0

        with self._workers_lock:
            for wid, sock in self._workers.items():
                payload = (W0, b0)
                _send_msg(
                    sock,
                    {
                        "type": "step",
                        "payload": payload,
                        "seed": seed,
                        "start": start,
                        "n_batches": len(self._workers),
                        "batch_size": self.batch_size,
                    },
                )

                start += 1

    def step(self, model, X: np.ndarray, y: np.ndarray):
        """
        Ejecuta una época distribuida y actualiza los pesos del modelo.
        Devuelve (loss, acc, gnorm) con tiempos por sección.
        """
        t0 = time.perf_counter()

        n_workers = self._wait_workers()

        if n_workers is None:
            log.warning("Ningún worker conectado, saltando época")
            return 0, 0, 0

        layers = model.network.trainable_layers()
        W0 = [l.weights for l in layers]
        b0 = [l.bias for l in layers]

        self._broadcast_weights(W0, b0)

        results = self._collect_results()

        if not results:
            log.warning("Ningún worker devolvió resultado, saltando época")
            return 0, 0, 0

        gW_acc = [np.zeros_like(w) for w in W0]
        gb_acc = [np.zeros_like(b) for b in b0]
        total_loss = 0.0

        for grads_W, grads_b, loss in results:
            for i in range(len(gW_acc)):
                gW_acc[i] += grads_W[i]
                gb_acc[i] += grads_b[i]
            total_loss += loss

        m = len(results)
        for i, layer in enumerate(layers):
            layer.d_weights = gW_acc[i] / m
            layer.d_bias = gb_acc[i] / m

        gnorm = float(model._compute_grad_norm())
        model.optimizer.step(layers)

        sample_size = min(10000, len(X))  # asegurar no superar tamaño
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]

        y_pred_new = model.network.forward(X_sample, training=False)
        acc = float(model._accuracy(y_sample, y_pred_new))

        # Calcular throughput
        elapsed_time = time.perf_counter() - t0  # tiempo de la época
        throughput = sample_size / elapsed_time

        # Guardar métricas en el dataframe
        self.metrics.loc[len(self.metrics)] = [
            self._epoch_seed,  # epoch
            elapsed_time,  # time
            total_loss / m,  # total_loss
            acc,  # accuracy
            gnorm,  # grad norm
            throughput,  # throughput
            n_workers,  # n_workers
        ]
        return total_loss / m, acc, gnorm


# %% [markdown]
# ## Client

# %%

# ══════════════════════════════════════════════════════════════════════════════
# ABC — Cliente
# ══════════════════════════════════════════════════════════════════════════════


class ClientStrategy(ABC):
    """
    Base para el lado cliente (worker) del entrenamiento distribuido.

    Gestiona:
      - Conexión / reconexión TCP
      - Loop de mensajes
      - Despacho a handlers por tipo de mensaje

    Subclases implementan `run()` que define qué hacer con cada mensaje.
    """

    RECONNECT_DELAY = 3  # segundos entre intentos de reconexión
    RECONNECT_ATTEMPTS = 20  # intentos antes de rendirse

    def __init__(self, server_host: str, server_port: int):
        self.server_host = server_host
        self.server_port = server_port

        self._sock: Optional[socket.socket] = None
        self._worker_id: Optional[int] = None
        self._assignment: dict = {}
        self._connected = False

    def connect(self) -> None:
        """Conecta al servidor y recibe la asignación inicial."""
        for attempt in range(1, self.RECONNECT_ATTEMPTS + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )  # Reuse address
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1
                )  # Enable keepalive
                sock.connect((self.server_host, self.server_port))
                self._sock = sock

                # Enviar ready (con worker_id si es reconexión)
                msg: dict = {"type": "ready"}

                if self._worker_id is not None:
                    msg["worker_id"] = self._worker_id

                _send_msg(sock, msg)

                # Recibir asignación
                assign = _recv_msg(sock)

                if assign.get("type") != "assign":
                    raise ValueError(f"Esperaba 'assign', recibí {assign.get('type')}")

                self._worker_id = assign["worker_id"]
                self._assignment = {
                    k: v for k, v in assign.items() if k not in ("type", "worker_id")
                }
                self._connected = True

                log.info(
                    f"Conectado como worker {self._worker_id} | "
                    f"asignación: {self._assignment}"
                )
                return

            except Exception as e:
                log.warning(f"Intento {attempt}/{self.RECONNECT_ATTEMPTS} fallido: {e}")
                time.sleep(self.RECONNECT_DELAY)

        raise ConnectionError(
            f"No se pudo conectar a {self.server_host}:{self.server_port} "
            f"tras {self.RECONNECT_ATTEMPTS} intentos"
        )

    def close(self) -> None:
        """Cierra el socket limpiamente."""
        self._connected = False

        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
            except Exception:
                pass

            self._sock = None

        log.info("Cliente cerrado")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _reconnect(self) -> None:
        self.close()
        log.info("Intentando reconexión…")
        self.connect()

    def _loop(self, handlers: dict) -> None:
        """
        Loop principal de mensajes.
        `handlers` es un dict  tipo → callable(msg) → None
        El callable puede enviar de vuelta al servidor con _send_msg(self._sock, …)
        """
        while True:
            try:
                msg = _recv_msg(self._sock)
                mtype = msg.get("type")

                if mtype == "done":
                    log.info("Servidor indicó fin — cerrando")
                    break

                if mtype == "assign":
                    # re-asignación durante reconexión
                    self._assignment = {
                        k: v for k, v in msg.items() if k not in ("type", "worker_id")
                    }
                    log.info(f"Re-asignación recibida: {self._assignment}")
                    continue

                handler = handlers.get(mtype)

                if handler:
                    handler(msg)
                else:
                    log.warning(f"Mensaje desconocido ignorado: {mtype}")

            except (ConnectionError, OSError) as e:
                log.warning(f"Conexión perdida: {e}")

                try:
                    self._reconnect()
                except ConnectionError:
                    log.error("No se pudo reconectar — terminando cliente")
                    break

    @abstractmethod
    def run(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """Blocking loop — escucha mensajes y ejecuta la estrategia."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Estrategia concreta — Cliente: Batch Averaging
# ══════════════════════════════════════════════════════════════════════════════


class DistributedWeightAvgStrategy(ClientStrategy):
    """
    Worker para ServerWeightAvgStrategy.

    Mensajes que maneja:
      weights → carga W0, b0 en el modelo
      step    → forward + backward + optimizer.step → envía resultado

    Slice sin solapamiento
    ──────────────────────
    Usa el seed y start que envió el servidor:
        rng  = np.random.default_rng(seed)
        idx  = rng.permutation(N)
        lo   = start * batch_size
        hi   = lo + batch_size          (o N si es el último)
        X_b  = X[idx[lo:hi]]
    """

    def __init__(self, server_host: str, server_port: int):
        super().__init__(server_host, server_port)
        self._W0: Optional[list] = None
        self._b0: Optional[list] = None
        self._model = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    # ── Handlers ──────────────────────────────────────────────────────────

    def _handle_weights(self, msg: dict) -> None:
        W0, b0 = msg["payload"]
        self._W0 = W0
        self._b0 = b0
        log.debug("Pesos W0 recibidos")

    def _handle_step(self, _msg: dict) -> None:
        """Ejecuta un paso de entrenamiento y envía resultado al servidor."""
        if self._W0 is None:
            log.error("step recibido antes de weights — ignorando")
            return

        model = self._model
        X, y = self._X, self._y
        N = len(X)

        # Extraer slice sin solapamiento
        seed = self._assignment.get("seed", 0)
        start = self._assignment.get("start", 0)
        n_slices = self._assignment.get("n_batches", 1)

        # batch_size = max(1, N // n_slices)
        batch_size = self._assignment.get("batch_size", 0)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        lo = start * batch_size
        hi = lo + batch_size if start < n_slices - 1 else N

        X_b = X[idx[lo:hi]]
        y_b = y[idx[lo:hi]]

        layers = model.network.trainable_layers()

        # Cargar W0
        for i, layer in enumerate(layers):
            layer.weights = self._W0[i].copy()
            layer.bias = self._b0[i].copy()

        # Forward / backward / step
        y_pred = model.network.forward(X_b, training=True)
        loss = float(model.cost.function(y_b, y_pred))
        delta = model.cost.derivative(y_b, y_pred)
        model.network.backward(delta)
        model.optimizer.step(layers)

        acc = float(model._accuracy(y_b, y_pred))
        gnorm = float(model._compute_grad_norm())

        Wb = [l.weights.copy() for l in layers]
        bb = [l.bias.copy() for l in layers]

        _send_msg(
            self._sock,
            {
                "type": "result",
                "worker_id": self._worker_id,
                "payload": (Wb, bb, loss, acc, gnorm),
            },
        )
        log.debug(f"Resultado enviado | loss={loss:.4f} acc={acc:.4f}")

    # ── run ───────────────────────────────────────────────────────────────

    def run(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """
        Blocking loop.  Escucha mensajes del servidor y ejecuta el paso
        correspondiente.

        El `model` solo se usa para acceder a network / cost / optimizer /
        _accuracy / _compute_grad_norm.  El optimizer.step() se llama aquí
        (no en el servidor), igual que ParallelWeightAvgStrategy.
        """
        self._model = model
        self._X = X
        self._y = y

        self._loop(
            {
                "weights": self._handle_weights,
                "step": self._handle_step,
            }
        )

        log.info("Cliente finalizado")


class DistributedGradientAvgStrategy(ClientStrategy):
    """
    Mensajes que maneja:
      weights → carga W0, b0 en el modelo
      step    → forward + backward → envía resultado
    """

    def __init__(self, server_host: str, server_port: int):
        super().__init__(server_host, server_port)
        self._W0: Optional[list] = None
        self._b0: Optional[list] = None
        self._model = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

        self._local_epoch = 0

        self.metrics = pd.DataFrame(
            columns=[
                "worker_id",
                "local_epoch",
                "time",
                "loss",
                "gnorm",
                "throughput",
                "n_batches",
                "batch_size",
                "seed",
                "start",
            ]
        )

    # ── Handlers ──────────────────────────────────────────────────────────

    def _load_data(self, msg: dict) -> None:
        W0, b0 = msg["payload"]
        self._W0 = W0
        self._b0 = b0
        seed = msg.get("seed", 0)
        start = msg.get("start", 0)
        n_batches = msg.get("n_batches", 0)
        batch_size = msg.get("batch_size", 0)
        self._local_epoch += 1
        self._run_id = int(time.time())

        self._assignment["seed"] = seed
        self._assignment["start"] = start
        self._assignment["n_batches"] = n_batches
        self._assignment["batch_size"] = batch_size

        log.debug(f"Pesos W0 recibidos | {seed=} | {start=} {n_batches=} {batch_size=}")

    @time_wrapper
    def _handle_step(self, _msg: dict) -> None:
        """Ejecuta un paso de entrenamiento y envía resultado al servidor."""
        t0 = time.perf_counter()
        self._load_data(_msg)

        if self._W0 is None:
            log.error("step recibido antes de weights — ignorando")
            return

        model = self._model
        X, y = self._X, self._y
        N = len(X)

        seed = self._assignment["seed"]
        start = self._assignment["start"]
        n_batches = self._assignment["n_batches"]
        batch_size = self._assignment["batch_size"]

        # Cada worker obtiene su slice exclusivo (sin solapamiento)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        lo = start * batch_size
        hi = lo + batch_size if start < n_batches - 1 else N

        X_b = X[idx[lo:hi]]
        y_b = y[idx[lo:hi]]

        layers = model.network.trainable_layers()

        # Cargar W0
        for i, layer in enumerate(layers):
            layer.weights = self._W0[i]
            layer.bias = self._b0[i]

        # Forward / backward
        y_pred = model.network.forward(X_b, training=True)
        loss = float(model.cost.function(y_b, y_pred))
        delta = model.cost.derivative(y_b, y_pred)
        model.network.backward(delta)

        grads_W = [l.d_weights for l in layers]
        grads_b = [l.d_bias for l in layers]

        _send_msg(
            self._sock,
            {
                "type": "result",
                "worker_id": self._worker_id,
                "payload": (grads_W, grads_b, loss),
            },
        )
        log.debug(f"Resultado enviado | loss={loss:.4f}")

        t1 = time.perf_counter()
        elapse_time = t1 - t0

        self.metrics.loc[len(self.metrics)] = [
            self._worker_id,
            self._local_epoch,
            elapse_time,
            loss,
            model._compute_grad_norm(),
            len(X_b) / elapse_time,
            n_batches,
            batch_size,
            seed,
            start,
        ]

    # ── run ───────────────────────────────────────────────────────────────

    def run(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """
        Blocking loop.  Escucha mensajes del servidor y ejecuta el paso
        correspondiente.
        """
        self._model = model
        self._X = X
        self._y = y

        try:
            self._loop(
                {
                    "step": self._handle_step,
                }
            )
        finally:
            # Garantiza cierre del socket aunque run() termine por excepcion
            filename = (
                f"metrics_worker{self._worker_id}"
                f"_run{self._run_id}"
                f"_{time.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.metrics.to_csv(filename)
            self.close()

        log.info("Cliente finalizado")


# %% [markdown]
# ## Run Server/Client


def run_local(workers: int = 1, lr=0.05, epochs=200):
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))

    strategy = GradientAvgStrategy(batch_size=len(X_train) // workers)

    save_folder = f"local_metrics__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        model = Model(net, MeanSquaredError(), SGD(lr), strategy=strategy)
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose_epoch=25,
            X_val=X_test,
            y_val=y_test,
        )
        history.set_output_dir(save_folder)
        history.plot()
        model.evaluate(X_test, y_test)
        model.confusion_matrix(X_test, y_test)
        model.classification_report(X_test, y_test)
        history.save_all()
    except Exception as e:
        print(e)
    finally:
        strategy.metrics.to_csv(f"{save_folder}/metrics_server.csv", index=False)

        # Guardar descripción de métricas
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "epochs": epochs,
                    "workers": workers,
                },
                f,
            )


# %%
def run_server(
    server_host: str,
    server_port: int,
    workers: int = 1,
    min_workers: int = 1,
    lr=0.05,
    epochs=200,
):
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))

    strategy = ServerGradientAvgStrategy(
        batch_size=len(X_train) // workers, min_workers=min_workers
    )
    strategy.start_server(server_port, server_host)

    save_folder = f"metrics__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        model = Model(net, MeanSquaredError(), SGD(lr), strategy=strategy)
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose_epoch=25,
            X_val=X_test,
            y_val=y_test,
        )
        history.set_output_dir(save_folder)
        history.plot()
        model.evaluate(X_test, y_test)
        model.confusion_matrix(X_test, y_test)
        model.classification_report(X_test, y_test)
        history.save_all()
    except Exception as e:
        print(e)
    finally:
        strategy.stop_server()
        strategy.metrics.to_csv(f"{save_folder}/metrics_server.csv", index=False)

        # Guardar descripción de métricas
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "epochs": epochs,
                    "workers": workers,
                    "min_workers": min_workers,
                },
                f,
            )


# %%
def run_client(server_host: str, server_port: int, lr=float):
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))

    model = Model(net, MeanSquaredError(), SGD(lr))
    strategy = DistributedGradientAvgStrategy(server_host, server_port)

    save_folder = f"metrics_client__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        strategy.connect()
        strategy.run(model, X_train, y_train)
    except Exception as e:
        print(e)
        strategy.close()
    finally:
        # Guardar métricas locales del worker
        strategy.metrics.to_csv(f"{save_folder}/metrics_client.csv", index=False)

        # Guardar la descripción de las métricas (stats generales)
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # Guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "server_host": server_host,
                    "server_port": server_port,
                },
                f,
            )


def main():
    # arg 1 --server | --client
    # arg 2 --host
    # arg 3 --port
    #
    # ejemplo:
    # python mnist.py --server --host 0.0.0.0 --port 9999 --workers 2 --min_workers 1
    # python mnist.py --client --host 0.0.0.0 --port 9999
    #
    print_system_info()

    parser = argparse.ArgumentParser(description="Distributed Gradient Averaging")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9999, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--min_workers", default=1, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--epochs", default=200, type=int)

    args = parser.parse_args()
    print(args)

    if args.local:
        run_local(args.workers, args.lr, args.epochs)
    elif args.server:
        run_server(
            args.host, args.port, args.workers, args.min_workers, args.lr, args.epochs
        )
    elif args.client:
        run_client(args.host, args.port, args.lr)
    else:
        raise ValueError("No se especificó --local, --server o --client")

    return 0


if __name__ == "__main__":
    main()
