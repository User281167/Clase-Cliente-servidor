from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from nncore.distributed import ClientStrategy
from nncore.distributed.utils import _send_msg, log
from nncore.utils import time_wrapper


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
            ]
        )

    # ── Handlers ──────────────────────────────────────────────────────────

    def _handle_weights(self, msg: dict) -> None:
        W0, b0 = msg["payload"]
        self._W0 = W0
        self._b0 = b0

    @time_wrapper
    def _handle_step(self, _msg: dict) -> None:
        t0 = time.perf_counter()

        if self._W0 is None:
            log.error("step recibido antes de weights")
            return

        model = self._model
        dataset = self._dataset
        N = len(dataset)

        rank = self._assignment["rank"]
        world_size = self._assignment["world_size"]
        epoch = self._assignment["epoch"]
        batch_size = self._assignment["batch_size"]

        # DISTRIBUTED SAMPLER tomar los datos del worker actual
        # Permutation igual para
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(N)
        indices = indices[rank::world_size]

        batch_idx = indices[:batch_size]
        X_b, y_b = dataset[batch_idx]

        layers = model.network.trainable_layers()

        # cargar pesos
        for i, layer in enumerate(layers):
            layer.weights = self._W0[i]
            layer.bias = self._b0[i]

        # forward / backward
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

        t1 = time.perf_counter()
        elapsed = t1 - t0

        self.metrics.loc[len(self.metrics)] = [
            self._worker_id,
            epoch,
            elapsed,
            loss,
            model.compute_grad_norm(),
            len(X_b) / elapsed,
        ]

    # ── run ───────────────────────────────────────────────────────────────

    def run(self, model, dataset) -> None:
        """
        Blocking loop.  Escucha mensajes del servidor y ejecuta el paso
        correspondiente.
        """
        self._model = model
        self._dataset = dataset

        try:
            self._loop(
                {
                    "weights": self._handle_weights,
                    "step": self._handle_step,
                }
            )
        except Exception as e:
            log.exception(e)
        finally:
            # Garantiza cierre del socket aunque run() termine por excepcion
            filename = (
                f"metrics_worker{self._worker_id}"
                f"_run{self._assignment.get('rank', 0)}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.metrics.to_csv(filename)
            self.close()

        log.info("Cliente finalizado")
