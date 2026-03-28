import numpy as np
from numpy.lib.stride_tricks import as_strided

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: im2col / col2im
# ──────────────────────────────────────────────────────────────────────────────
#
#  im2col  (versión rápida con as_strided)
#  ──────────────────────────────────────
#  Convierte un tensor de imágenes (N, C, H, W) en una matriz de columnas
#  donde cada columna es un parche aplanado de shape (C*kH*kW).
#
#  Implementación con np.lib.stride_tricks.as_strided:
#  En lugar de un loop Python sobre (H_out, W_out), se construye una VISTA
#  del array reinterpretando los strides de memoria — sin copiar datos.
#
#    Vista intermedia shape: (N, C, H_out, W_out, kH, kW)
#    Strides: los dos ejes espaciales de salida avanzan stride*s_h / stride*s_w
#             bytes, los dos ejes del kernel avanzan s_h / s_w bytes normalmente.
#
#  Luego se transpone y reshapea a (N, C*kH*kW, H_out*W_out).
#  Cero loops Python — ~60x más rápido que la versión con loop.
#
#  ADVERTENCIA: as_strided produce una vista no contigua. Se llama
#  np.ascontiguousarray() al final para que el tensordot posterior
#  opere sobre memoria contigua y no sufra penalizaciones de caché.
#
#  col2im
#  ──────
#  Operación inversa: acumula gradientes de X_col de vuelta al tensor
#  original (N, C, H, W). Usa np.add.at para acumulación vectorizada
#  sin loops Python sobre posiciones espaciales.
#
#  Cada posición puede haber contribuido a múltiples parches
#  (solapamiento cuando stride < kernel_size), por lo que los
#  gradientes se SUMAN con += (no se asignan con =).


def im2col(
    X_padded: np.ndarray, kH: int, kW: int, stride: int, H_out: int, W_out: int
) -> np.ndarray:
    """
    Extrae todos los parches de X_padded en una sola operación vectorizada.

    Usa as_strided para construir una vista (N, C, H_out, W_out, kH, kW)
    reinterpretando los strides de memoria — sin copiar ni loopar.

    Parámetros
    ----------
    X_padded : (N, C, H_pad, W_pad)  — imagen con padding ya aplicado
    kH, kW   : tamaño del filtro
    stride   : paso de la ventana
    H_out, W_out : dimensiones espaciales de la salida

    Retorna
    -------
    X_col : (N, C*kH*kW, H_out*W_out)  — contiguous array listo para matmul
    """
    N, C, H_pad, W_pad = X_padded.shape
    s_n, s_c, s_h, s_w = X_padded.strides

    # Vista que expone cada parche sin copiar datos
    # Los ejes (H_out, W_out) avanzan stride*s_h / stride*s_w bytes
    # Los ejes (kH, kW) avanzan s_h / s_w bytes (paso normal del array)
    shape = (N, C, H_out, W_out, kH, kW)
    strides = (s_n, s_c, stride * s_h, stride * s_w, s_h, s_w)
    patches = as_strided(X_padded, shape=shape, strides=strides)

    # (N, C, H_out, W_out, kH, kW)
    # → transponer a (N, C, kH, kW, H_out, W_out)
    # → reshape  a  (N, C*kH*kW, H_out*W_out)
    X_col = patches.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * kH * kW, H_out * W_out)

    # ascontiguousarray garantiza memoria contigua para el tensordot posterior
    return np.ascontiguousarray(X_col)  # (N, C*kH*kW, H_out*W_out)


def col2im(
    dX_col: np.ndarray,
    input_shape: tuple,
    kH: int,
    kW: int,
    stride: int,
    padding: int,
    H_out: int,
    W_out: int,
) -> np.ndarray:
    """
    Acumula gradientes de dX_col de vuelta al tensor original.

    Loop sobre H_out*W_out posiciones — cada iteración es vectorizada
    sobre (N, C, kH, kW). No existe equivalente con as_strided porque
    la operación requiere acumulación con +=, no solo lectura.

    Parámetros
    ----------
    dX_col     : (N, C*kH*kW, H_out*W_out)  gradiente respecto a X_col
    input_shape: (N, C, H, W)                shape ORIGINAL (sin padding)
    kH, kW     : tamaño del filtro
    stride     : paso de la ventana
    padding    : padding aplicado en forward
    H_out, W_out : dimensiones espaciales de la salida

    Retorna
    -------
    dX : (N, C, H, W)  gradiente respecto a la entrada original
    """
    N, C, H, W = input_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    dX_padded = np.zeros((N, C, H_pad, W_pad), dtype=dX_col.dtype)

    col = 0
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            # (N, C*kH*kW) → (N, C, kH, kW), luego acumular
            dX_padded[:, :, h_start : h_start + kH, w_start : w_start + kW] += dX_col[
                :, :, col
            ].reshape(N, C, kH, kW)
            col += 1

    if padding > 0:
        return dX_padded[:, :, padding:-padding, padding:-padding]

    return dX_padded  # (N, C, H, W)
