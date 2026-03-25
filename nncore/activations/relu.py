import numpy as np

from .activation import ActivationFunction


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
