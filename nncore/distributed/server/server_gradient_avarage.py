import time

import numpy as np
import pandas as pd

from nncore.distributed.utils import _send_msg, log

from .server_strategy import ServerTrainingStrategy


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

        gnorm = float(model.compute_grad_norm())
        model.optimizer.step(layers)

        sample_size = min(10000, len(X))  # asegurar no superar tamaño
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]

        y_pred_new = model.network.forward(X_sample, training=False)
        acc = float(model.accuracy(y_sample, y_pred_new))

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
