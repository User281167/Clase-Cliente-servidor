import numpy as np

from nncore.activations import ActivationFunction
from nncore.initializers import WeightInitializer

from ._img_col_2 import col2im, im2col
from .layer import Layer

# ──────────────────────────────────────────────────────────────────────────────
# Conv2D
# ──────────────────────────────────────────────────────────────────────────────
#
#  FORWARD
#  ───────
#  Entrada X shape (N, C_in, H, W)
#
#  1) Aplicar padding:
#
#       X_padded  shape: (N, C_in, H+2p, W+2p)
#
#  2) Calcular dimensiones de salida:
#
#       H_out = (H + 2*padding - kH) // stride + 1
#       W_out = (W + 2*padding - kW) // stride + 1
#
#  3) im2col — convertir parches en columnas:
#
#       X_col  shape: (N, C_in*kH*kW, H_out*W_out)
#
#  4) Aplanar filtros:
#
#       W_col  shape: (C_out, C_in*kH*kW)
#
#  5) Matmul por muestra + bias:
#
#       Z[n] = W_col @ X_col[n] + b   shape: (C_out, H_out*W_out)
#       Z    shape final: (N, C_out, H_out, W_out)
#
#  6) Activación:
#
#       A = f(Z)                       shape: (N, C_out, H_out, W_out)
#
#  BACKWARD
#  ────────
#  Recibe delta = ∂L/∂A  shape (N, C_out, H_out, W_out)
#
#  1) Gradiente pre-activación:
#
#       dZ = delta ⊙ f'(Z)             shape: (N, C_out, H_out, W_out)
#
#  2) Aplanar dZ para matmul:
#
#       dZ_col[n] shape: (C_out, H_out*W_out)
#
#  3) Gradiente de los filtros (acumulado sobre N):
#
#       ∂L/∂W = Σ_n  dZ_col[n] @ X_col[n].T   shape: (C_out, C_in*kH*kW)
#             → reshape → (C_out, C_in, kH, kW)
#
#  4) Gradiente del bias:
#
#       ∂L/∂b = sum(dZ, axis=(0,2,3))          shape: (C_out, 1, 1)
#
#  5) Gradiente respecto a X_col:
#
#       dX_col[n] = W_col.T @ dZ_col[n]        shape: (C_in*kH*kW, H_out*W_out)
#
#  6) col2im — acumular dX_col de vuelta al tensor original:
#
#       ∂L/∂X = col2im(dX_col)                 shape: (N, C_in, H, W)


class Conv2D(Layer):
    """
    Capa convolucional 2D con im2col.

    Espera entrada en formato NCHW: (N, C_in, H, W).

    Parámetros
    ----------
    in_channels  : C_in — canales de entrada (1 para grayscale, 3 para RGB)
    out_channels : C_out — número de filtros a aprender
    kernel_size  : tamaño del filtro cuadrado (kH = kW = kernel_size)
    stride       : paso de la ventana deslizante (default 1)
    padding      : píxeles de ceros a añadir en cada borde (default 0)
    activation   : instancia de ActivationFunction (ej. ReLU())
    weight_initializer : instancia de WeightInitializer (ej. HeUniform())

    Pesos
    -----
    weights : (C_out, C_in, kH, kW)   — filtros
    bias    : (C_out, 1, 1)            — un bias por filtro, broadcast sobre H y W

    Estado guardado en forward (necesario para backward)
    ────────────────────────────────────────────────────
    _input_shape : (N, C_in, H, W)
    X_padded     : (N, C_in, H+2p, W+2p)
    X_col        : (N, C_in*kH*kW, H_out*W_out)
    Z            : (N, C_out, H_out, W_out)  pre-activación
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: ActivationFunction | None = None,
        weight_initializer: WeightInitializer | None = None,
    ):
        # La clase base usa input_size/output_size para Dense.
        # Para Conv2D los pasamos como None — los pesos se inicializan aparte.
        super().__init__(
            input_size=None,
            output_size=None,
            activation=activation,
            weight_initializer=None,  # inicializamos W manualmente abajo
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = stride
        self.padding = padding

        # Pesos: (C_out, C_in, kH, kW)
        # Se inicializan tratando (C_in*kH*kW) como fan_in y C_out como fan_out
        if weight_initializer is not None:
            fan_in = in_channels * kernel_size * kernel_size
            fan_out = out_channels
            flat_W = weight_initializer.initialize(fan_in, fan_out)  # (fan_in, fan_out)
            self.weights = flat_W.T.reshape(
                out_channels, in_channels, kernel_size, kernel_size
            )
        else:
            # fallback: He uniforme manual
            fan_in = in_channels * kernel_size * kernel_size
            std = np.sqrt(2.0 / fan_in)
            self.weights = (
                np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
                * std
            )

        # Bias: (C_out, 1, 1) — hace broadcast sobre H_out y W_out
        self.bias = np.zeros((out_channels, 1, 1))

        # Estado del forward — se llena en cada llamada
        self._input_shape = None
        self.X_padded = None
        self.X_col = None
        self.Z = None

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Paso hacia adelante de la convolución.

        Parámetros
        ----------
        X : (N, C_in, H, W)

        Retorna
        -------
        A : (N, C_out, H_out, W_out)
        """
        N, C, H, W = X.shape
        self._input_shape = X.shape

        # 1) Padding
        if self.padding > 0:
            self.X_padded = np.pad(
                X,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            self.X_padded = X

        # 2) Dimensiones de salida
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1

        # 3) im2col: (N, C_in*kH*kW, H_out*W_out)
        self.X_col = im2col(self.X_padded, self.kH, self.kW, self.stride, H_out, W_out)

        # 4) Aplanar filtros: (C_out, C_in*kH*kW)
        W_col = self.weights.reshape(self.out_channels, -1)

        # 5) Convolución como matmul + bias
        #    Z[n] = W_col @ X_col[n]  →  (C_out, H_out*W_out)
        #    np.tensordot o loop sobre N — loop es más claro aquí
        Z_col = np.tensordot(W_col, self.X_col, axes=([1], [1]))

        # Z_col shape: (C_out, N, H_out*W_out) → (N, C_out, H_out*W_out)
        Z_col = Z_col.transpose(1, 0, 2)
        self.Z = Z_col.reshape(N, self.out_channels, H_out, W_out) + self.bias

        # 6) Activación
        if self.activation is not None:
            return self.activation(self.Z)

        return self.Z

    # ──────────────────────────────────────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────────────────────────────────────

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Paso hacia atrás de la convolución.

        Parámetros
        ----------
        delta : ∂L/∂A  shape (N, C_out, H_out, W_out)

        Retorna
        -------
        ∂L/∂X  shape (N, C_in, H, W)
        """
        N, C_out, H_out, W_out = delta.shape

        # 1) Gradiente pre-activación
        if self.activation is not None:
            dZ = delta * self.activation.derivative(self.Z)  # (N, C_out, H_out, W_out)
        else:
            dZ = delta

        # 2) Aplanar dZ: (N, C_out, H_out*W_out)
        dZ_col = dZ.reshape(N, C_out, -1)

        # 3) ∂L/∂W  (acumulado sobre N)
        #    dW_col = Σ_n  dZ_col[n] @ X_col[n].T
        #    dZ_col[n]: (C_out, H_out*W_out)
        #    X_col[n] : (C_in*kH*kW, H_out*W_out)
        #    resultado: (C_out, C_in*kH*kW)
        dW_col = np.tensordot(dZ_col, self.X_col, axes=([0, 2], [0, 2]))
        self.d_weights = dW_col.reshape(self.weights.shape)  # (C_out, C_in, kH, kW)

        # 4) ∂L/∂b  — suma sobre N, H_out, W_out
        self.d_bias = dZ.sum(axis=(0, 2, 3), keepdims=False).reshape(C_out, 1, 1)

        # 5) ∂L/∂X_col
        #    W_col.T @ dZ_col[n]
        #    W_col  : (C_out, C_in*kH*kW)
        #    dZ_col : (N, C_out, H_out*W_out)
        #    resultado dX_col: (N, C_in*kH*kW, H_out*W_out)
        W_col = self.weights.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        dX_col = np.tensordot(W_col.T, dZ_col, axes=([1], [1]))
        # shape: (C_in*kH*kW, N, H_out*W_out) → (N, C_in*kH*kW, H_out*W_out)
        dX_col = dX_col.transpose(1, 0, 2)

        # 6) col2im — acumular gradientes al tensor original
        return col2im(
            dX_col,
            self._input_shape,
            self.kH,
            self.kW,
            self.stride,
            self.padding,
            H_out,
            W_out,
        )  # (N, C_in, H, W)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self):
        act = self.activation.__class__.__name__ if self.activation else "None"
        return (
            f"Conv2D("
            f"in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"kernel={self.kH}×{self.kW}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"activation={act})"
        )

    def compute_output_shape(self, input_shape):
        C, H, W = input_shape
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        return (self.out_channels, H_out, W_out)
