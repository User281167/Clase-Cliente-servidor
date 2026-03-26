from __future__ import annotations

from typing import Optional

import numpy as np

from nncore import Model
from nncore.distributed.utils import _send_msg, log

from .client_strategy import ClientStrategy


class DistributedWeightAvgStrategy(ClientStrategy):
    """
    Worker para ServerWeightAvgStrategy.

    Mensajes que maneja:
      weights → carga W0, b0 en el modelo
      step    → forward + backward + optimizer.step → envía resultado

    Slice sin solapamiento
    ──────────────────────
    Usa el seed y start que envió el servidor:
        rng  = np.random.default_rng(seed)
        idx  = rng.permutation(N)
        lo   = start * batch_size
        hi   = lo + batch_size          (o N si es el último)
        X_b  = X[idx[lo:hi]]
    """

    def __init__(self, server_host: str, server_port: int):
        super().__init__(server_host, server_port)
        self._W0: Optional[list] = None
        self._b0: Optional[list] = None
        self._model: Model | None = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    # ── Handlers ──────────────────────────────────────────────────────────

    def _handle_weights(self, msg: dict) -> None:
        W0, b0 = msg["payload"]
        self._W0 = W0
        self._b0 = b0
        log.debug("Pesos W0 recibidos")

    def _handle_step(self, _msg: dict) -> None:
        """Ejecuta un paso de entrenamiento y envía resultado al servidor."""
        if self._W0 is None:
            log.error("step recibido antes de weights — ignorando")
            return

        model = self._model
        X, y = self._X, self._y
        N = len(X)

        # Extraer slice sin solapamiento
        seed = self._assignment.get("seed", 0)
        start = self._assignment.get("start", 0)
        n_slices = self._assignment.get("n_batches", 1)

        # batch_size = max(1, N // n_slices)
        batch_size = self._assignment.get("batch_size", 0)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        lo = start * batch_size
        hi = lo + batch_size if start < n_slices - 1 else N

        X_b = X[idx[lo:hi]]
        y_b = y[idx[lo:hi]]

        layers = model.network.trainable_layers()

        # Cargar W0
        for i, layer in enumerate(layers):
            layer.weights = self._W0[i].copy()
            layer.bias = self._b0[i].copy()

        # Forward / backward / step
        y_pred = model.network.forward(X_b, training=True)
        loss = float(model.cost.function(y_b, y_pred))
        delta = model.cost.derivative(y_b, y_pred)
        model.network.backward(delta)
        model.optimizer.step(layers)

        acc = float(model.accuracy(y_b, y_pred))
        gnorm = float(model.compute_grad_norm())

        Wb = [l.weights.copy() for l in layers]
        bb = [l.bias.copy() for l in layers]

        _send_msg(
            self._sock,
            {
                "type": "result",
                "worker_id": self._worker_id,
                "payload": (Wb, bb, loss, acc, gnorm),
            },
        )
        log.debug(f"Resultado enviado | loss={loss:.4f} acc={acc:.4f}")

    # ── run ───────────────────────────────────────────────────────────────

    def run(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """
        Blocking loop.  Escucha mensajes del servidor y ejecuta el paso
        correspondiente.

        El `model` solo se usa para acceder a network / cost / optimizer /
        accuracy / compute_grad_norm.  El optimizer.step() se llama aquí
        (no en el servidor), igual que ParallelWeightAvgStrategy.
        """
        self._model = model
        self._X = X
        self._y = y

        self._loop(
            {
                "weights": self._handle_weights,
                "step": self._handle_step,
            }
        )

        log.info("Cliente finalizado")
