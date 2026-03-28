import numpy as np
from numpy.lib.stride_tricks import as_strided

from .layer import Layer

# ──────────────────────────────────────────────────────────────────────────────
# MaxPool2D  —  max pooling espacial 2D
# ──────────────────────────────────────────────────────────────────────────────
#
#  Reduce las dimensiones espaciales (H, W) tomando el valor máximo
#  de cada ventana de tamaño (pool_size × pool_size).
#
#  No tiene parámetros aprendibles — sin W, sin b, sin activación.
#  El único estado que se guarda es la MÁSCARA de posiciones ganadoras,
#  necesaria para redirigir los gradientes en el backward.
#
#  FORWARD
#  ───────
#  Entrada X shape (N, C, H, W)
#
#  Dimensiones de salida:
#
#       H_out = (H - pool_size) // stride + 1
#       W_out = (W - pool_size) // stride + 1
#
#  Para cada ventana de shape (pool_size, pool_size):
#
#       A[n, c, i, j] = max( X[n, c, i*s:i*s+p, j*s:j*s+p] )
#
#  Se guarda la máscara booleana del mismo shape que X:
#
#       mask[n, c, h, w] = True  si ese pixel fue el max de su ventana
#                        = False en caso contrario
#
#  BACKWARD
#  ────────
#  Recibe delta = ∂L/∂A  shape (N, C, H_out, W_out)
#
#  No hay pesos que actualizar — solo propagar el gradiente hacia atrás.
#  El gradiente fluye ÚNICAMENTE hacia el pixel que ganó el max en cada
#  ventana (derivada del max = 1 para el ganador, 0 para los demás):
#
#       ∂L/∂X[n,c,h,w] = delta[n,c,i,j]  si mask[n,c,h,w] == True
#                      = 0               si mask[n,c,h,w] == False
#
#  donde (i, j) es la posición en la salida que corresponde a la ventana
#  que contiene al pixel (h, w).
#
#  EMPATE
#  ──────
#  Si dos píxeles tienen el mismo valor máximo en una ventana, numpy
#  argmax() elige el primero en orden row-major. El gradiente va solo
#  a ese — el comportamiento es determinista y consistente entre
#  forward y backward.


class MaxPool2D(Layer):
    """
    pool_size : tamaño de la ventana cuadrada (pool_size × pool_size)
    stride    : paso entre ventanas. Si es None, se usa pool_size
                (ventanas no solapadas — el caso estándar).

    Shapes
    ------
    Entrada : (N, C, H, W)
    Salida  : (N, C, H_out, W_out)
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1

    Estado guardado en forward
    ──────────────────────────
    _input_shape : tuple  — (N, C, H, W) para reconstruir dX
    _mask        : ndarray bool (N, C, H, W) — posiciones ganadoras
    """

    def __init__(self, pool_size: int = 2, stride: int | None = None):
        # Sin input_size / output_size / activation / initializer
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self._input_shape = None
        self._flat_idx = None  # argmax en eje p² — shape (N, C, H_out, W_out)
        self._h_starts = None  # coord h de inicio por ventana — shape (H_out*W_out,)
        self._w_starts = None  # coord w de inicio por ventana — shape (H_out*W_out,)

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica max pooling y guarda la máscara de ganadores.

        Parámetros
        ----------
        X : (N, C, H, W)

        Retorna
        -------
        A : (N, C, H_out, W_out)
        """
        N, C, H, W = X.shape
        self._input_shape = X.shape
        p = self.pool_size
        s = self.stride

        H_out = (H - p) // s + 1
        W_out = (W - p) // s + 1

        # Vista de todos los parches sin copiar datos
        # shape: (N, C, H_out, W_out, p, p)
        s_n, s_c, s_h, s_w = X.strides
        patches = as_strided(
            X,
            shape=(N, C, H_out, W_out, p, p),
            strides=(s_n, s_c, s * s_h, s * s_w, s_h, s_w),
        )

        # Aplanar ventana y calcular max + argmax sobre el eje p²
        flat = patches.reshape(N, C, H_out, W_out, p * p)
        A = flat.max(axis=-1)  # (N, C, H_out, W_out)
        self._flat_idx = flat.argmax(axis=-1)  # (N, C, H_out, W_out)

        # Precalcular coordenadas de inicio de cada ventana en X
        # h_starts[i*W_out + j] = i*s,  w_starts[i*W_out + j] = j*s
        h_grid, w_grid = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing="ij")
        self._h_starts = (h_grid * s).ravel()  # (H_out*W_out,)
        self._w_starts = (w_grid * s).ravel()  # (H_out*W_out,)

        return A  # (N, C, H_out, W_out)

    # ──────────────────────────────────────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────────────────────────────────────
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Propaga el gradiente solo hacia los píxeles ganadores.

        Parámetros
        ----------
        delta : ∂L/∂A  shape (N, C, H_out, W_out)

        Retorna
        -------
        dX : (N, C, H, W)  — gradiente respecto a la entrada original
        """
        N, C, H, W = self._input_shape
        p = self.pool_size
        H_out, W_out = delta.shape[2], delta.shape[3]

        dX = np.zeros(self._input_shape, dtype=delta.dtype)

        # flat_idx → (row, col) dentro de la ventana de tamaño p×p
        row_in_win = self._flat_idx // p  # (N, C, H_out, W_out)
        col_in_win = self._flat_idx % p  # (N, C, H_out, W_out)

        # coord global del ganador en X
        h_starts = self._h_starts.reshape(H_out, W_out)
        w_starts = self._w_starts.reshape(H_out, W_out)
        h_global = h_starts[None, None, :, :] + row_in_win  # (N, C, H_out, W_out)
        w_global = w_starts[None, None, :, :] + col_in_win  # (N, C, H_out, W_out)

        # índices de batch y canal para indexing avanzado 4D
        n_idx = np.arange(N)[:, None, None, None]  # (N, 1, 1, 1)
        c_idx = np.arange(C)[None, :, None, None]  # (1, C, 1, 1)

        # np.add.at acumula correctamente si stride < pool_size (ventanas solapadas)
        np.add.at(dX, (n_idx, c_idx, h_global, w_global), delta)

        return dX  # (N, C, H, W)

    def __repr__(self):
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"

    def compute_output_shape(self, input_shape):
        C, H, W = input_shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        return (C, H_out, W_out)
