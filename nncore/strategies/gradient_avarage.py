import time

import numpy as np
import pandas as pd

from nncore.utils import get_batches

from .training_strategy import TrainingStrategy


class GradAvarageStrategy(TrainingStrategy):
    """
    Gradient Averaging — acumula gradientes de N workers,
    promedia, y hace UN SOLO paso del optimizer por época.

    Simula batch grande sin cargar todo en memoria.

    Algoritmo:
        for batch_b in M_batches:
            forward(X_b) → backward(delta_b)
            grad_acc += d_weights_b

        d_weights_avg = grad_acc / M
        optimizer.step(d_weights_avg)    ← UN solo paso por época

        Ejemplo:
            batch 1 → calcula ∂L_1/∂W
            batch 2 → calcula ∂L_2/∂W
            grad_avg = avg(∂L_1/∂W, ∂L_2/∂W)
            W_new = W - optimizer(grad_avg)   ← optimizer se aplica UNA vez sobre el promediado

        Diferencia clave vs _minibatch_step:
            minibatch  : optimizer.step() en CADA batch  (M pasos por época)
            grad_accum : optimizer.step() UNA vez por época (1 paso por época)

        Cuándo usar:
        - Cuando el batch_size real no cabe en memoria
            (simula batch grande con batches pequeños)
        - Para comparar convergencia con menos actualizaciones de pesos

        Relación matemática:
        Si la loss es lineal en los gradientes, acumular M batches
        y promediar ≈ entrenar con batch_size * M muestras de una vez.
        En la práctica (con BatchNorm, Dropout) hay diferencias sutiles.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

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

        layers = model.network.trainable_layers()
        grad_acc = [
            {"W": np.zeros_like(l.weights), "b": np.zeros_like(l.bias)} for l in layers
        ]

        epoch_loss = epoch_acc = 0.0
        n_batches = 0

        for X_b, y_b in get_batches(X, y, self.batch_size):
            y_pred = model.network.forward(X_b, training=True)
            loss = model.cost.function(y_b, y_pred)
            delta = model.cost.derivative(y_b, y_pred)
            model.network.backward(delta)

            # Acumular — NO llamar optimizer aquí
            for i, layer in enumerate(layers):
                grad_acc[i]["W"] += layer.d_weights
                grad_acc[i]["b"] += layer.d_bias

            epoch_loss += float(loss)
            epoch_acc += float(model.accuracy(y_b, y_pred))
            n_batches += 1

        # Promediar y aplicar UNA vez
        for i, layer in enumerate(layers):
            layer.d_weights = grad_acc[i]["W"] / n_batches
            layer.d_bias = grad_acc[i]["b"] / n_batches

        model.optimizer.step(layers)

        epoch_gnorm = model.compute_grad_norm()
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
