import time

import numpy as np
import pandas as pd

from nncore.distributed import ServerTrainingStrategy
from nncore.distributed.utils import _send_msg, log


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
        self._epoch = 0

        self.metrics = pd.DataFrame(
            columns=[
                "epoch",  # Época
                "time",  # Tiempo total de la época
                "total_loss",  # Pérdida total
                "gnorm",  # Norma de gradiente global
                "n_workers",  # Número de workers
            ]
        )

    def _send_assing(self, n_workers: int):
        # puede ser el rank, epoca etc
        for i, (wid, sock) in enumerate(self._workers.items()):
            with self._workers_lock:
                wids = list(self._workers.keys())

            for i, wid in enumerate(wids):
                msg = {
                    "type": "assign",
                    "rank": i,
                    "world_size": n_workers,
                    "epoch": self._epoch,
                    "batch_size": self.batch_size,
                }

                self._assignments[wid] = msg

                try:
                    _send_msg(self._workers[wid], msg)
                except Exception as e:
                    log.warning(f"No se pudo asignar slice a worker {wid}: {e}")

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
        self._send_assing(n_workers)
        self._broadcast_step()

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

        # Calcular throughput
        elapsed_time = time.perf_counter() - t0  # tiempo de la época
        self._epoch += 1

        # Guardar métricas en el dataframe
        self.metrics.loc[len(self.metrics)] = [
            self._epoch,  # epoch
            elapsed_time,  # time
            total_loss / m,  # total_loss
            gnorm,  # grad norm
            n_workers,  # n_workers
        ]
        return total_loss / m, 0, gnorm
