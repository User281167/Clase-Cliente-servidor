from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from nncore.distributed.utils import _send_msg, log
from nncore.utils import time_wrapper

from .client_strategy import ClientStrategy


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
                "n_batches",
                "batch_size",
                "seed",
                "start",
            ]
        )

    # ── Handlers ──────────────────────────────────────────────────────────

    def _load_data(self, msg: dict) -> None:
        W0, b0 = msg["payload"]
        self._W0 = W0
        self._b0 = b0
        seed = msg.get("seed", 0)
        start = msg.get("start", 0)
        n_batches = msg.get("n_batches", 0)
        batch_size = msg.get("batch_size", 0)
        self._local_epoch += 1
        self._run_id = int(time.time())

        self._assignment["seed"] = seed
        self._assignment["start"] = start
        self._assignment["n_batches"] = n_batches
        self._assignment["batch_size"] = batch_size

        log.debug(f"Pesos W0 recibidos | {seed=} | {start=} {n_batches=} {batch_size=}")

    @time_wrapper
    def _handle_step(self, _msg: dict) -> None:
        """Ejecuta un paso de entrenamiento y envía resultado al servidor."""
        t0 = time.perf_counter()
        self._load_data(_msg)

        if self._W0 is None:
            log.error("step recibido antes de weights — ignorando")
            return

        model = self._model
        X, y = self._X, self._y
        N = len(X)

        seed = self._assignment["seed"]
        start = self._assignment["start"]
        n_batches = self._assignment["n_batches"]
        batch_size = self._assignment["batch_size"]

        # Cada worker obtiene su slice exclusivo (sin solapamiento)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        lo = start * batch_size
        hi = lo + batch_size if start < n_batches - 1 else N

        X_b = X[idx[lo:hi]]
        y_b = y[idx[lo:hi]]

        layers = model.network.trainable_layers()

        # Cargar W0
        for i, layer in enumerate(layers):
            layer.weights = self._W0[i]
            layer.bias = self._b0[i]

        # Forward / backward
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
        log.debug(f"Resultado enviado | loss={loss:.4f}")

        t1 = time.perf_counter()
        elapse_time = t1 - t0

        self.metrics.loc[len(self.metrics)] = [
            self._worker_id,
            self._local_epoch,
            elapse_time,
            loss,
            model.compute_grad_norm(),
            len(X_b) / elapse_time,
            n_batches,
            batch_size,
            seed,
            start,
        ]

    # ── run ───────────────────────────────────────────────────────────────

    def run(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """
        Blocking loop.  Escucha mensajes del servidor y ejecuta el paso
        correspondiente.
        """
        self._model = model
        self._X = X
        self._y = y

        try:
            self._loop(
                {
                    "step": self._handle_step,
                }
            )
        finally:
            # Garantiza cierre del socket aunque run() termine por excepcion
            filename = (
                f"metrics_worker{self._worker_id}"
                f"_run{self._run_id}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.metrics.to_csv(filename)
            self.close()

        log.info("Cliente finalizado")
