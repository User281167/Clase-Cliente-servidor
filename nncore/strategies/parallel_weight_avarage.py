import os
from concurrent.futures import as_completed

import numpy as np
import psutil
from loky import get_reusable_executor

from .training_strategy import TrainingStrategy


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

        acc = model.accuracy(y_b, y_pred)
        gnorm = model.compute_grad_norm()

        # Devolver pesos finales
        Wb = [l.weights for l in layers]
        bb = [l.bias for l in layers]

        return (
            Wb,
            bb,
            float(loss),
            float(acc),
            float(gnorm),
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
            print("Todos los procesos fallaron en esta época")
            return 0, 0, 0, 0

        # Promedio final
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / n_batches
            layer.bias = b_acc[i] / n_batches

        return (epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm / n_batches)
