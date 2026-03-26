import numpy as np

from nncore.utils import get_batches

from .training_strategy import TrainingStrategy


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
        layers = model.network.trainable_layers()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]
        W_acc = [np.zeros_like(l.weights) for l in layers]
        b_acc = [np.zeros_like(l.bias) for l in layers]

        epoch_loss = epoch_acc = epoch_gnorm = 0.0
        n_batches = 0

        for X_b, y_b in get_batches(X, y, self.batch_size, self.shuffle):
            # Cada batch parte del mismo punto W_0
            for i, layer in enumerate(layers):
                layer.weights = W0[i].copy()
                layer.bias = b0[i].copy()

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
            epoch_acc += float(model.accuracy(y_b, y_pred))
            epoch_gnorm += model.compute_grad_norm()
            n_batches += 1

        # Cargar promedio de destinos
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / n_batches
            layer.bias = b_acc[i] / n_batches

        return epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm / n_batches
