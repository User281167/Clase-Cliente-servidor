import time

import numpy as np

from .server_strategy import ServerTrainingStrategy, _send_msg, log


class ServerWeightAvgStrategy(ServerTrainingStrategy):
    """
    Equivalente distribuido de ParallelWeightAvgStrategy.

    Algoritmo por época:
        1. Captura W₀ de la red local
        2. Asigna a cada worker su slice de datos (seed + start)
        3. Broadcast W₀
        4. Broadcast 'step'  → workers hacen forward+backward+optimizer
        5. Recoge (Wb, bb, loss, acc, gnorm) de cada worker
        6. W_new = promedio(W₁, …, W_M)
        7. Carga W_new en la red local

    Sin solapamiento de datos
    ─────────────────────────
    El servidor genera un seed global y asigna `start` distinto a cada worker.
    Cada worker hace:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        X_b = X[idx[start * batch_size : (start+1) * batch_size]]

    Así todos usan el mismo shuffle pero porciones distintas.
    """

    def __init__(self, batch_size: int = 256, min_workers: int = 1):
        super().__init__(min_workers=min_workers)
        self.batch_size = batch_size
        self._epoch_seed = 0  # incrementa cada época para variar el shuffle

    def _assign_slices(self, n_workers: int) -> int:
        """
        Genera seed y asigna start a cada worker.
        Retorna el seed usado.
        """
        seed = int(time.time()) ^ (self._epoch_seed * 0x9E3779B9)
        self._epoch_seed += 1

        with self._workers_lock:
            wids = list(self._workers.keys())

        for i, wid in enumerate(wids):
            msg = {
                "type": "assign",
                "worker_id": wid,
                "seed": seed,
                "start": i,
                "n_batches": n_workers,
                "batch_size": self.batch_size,
            }

            self._assignments[wid] = msg

            try:
                _send_msg(self._workers[wid], msg)
            except Exception as e:
                log.warning(f"No se pudo asignar slice a worker {wid}: {e}")

        return seed

    def step(self, model, X: np.ndarray, y: np.ndarray):
        """
        Ejecuta una época distribuida y actualiza los pesos del modelo.
        Devuelve (loss, acc, gnorm) — igual que cualquier TrainingStrategy.
        """
        n_workers = self._wait_workers()

        if n_workers is None:
            log.warning("Ningún worker conectado, saltando época")
            return 0, 0, 0

        layers = model.network.trainable_layers()
        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]

        # 1. Asignar slices
        self._assign_slices(n_workers)

        # 2. Broadcast pesos
        self._broadcast_weights(W0, b0)

        # 3. Broadcast step
        self._broadcast_step()

        # 4. Recoger resultados
        results = self._collect_results()

        if not results:
            log.warning("Ningún worker devolvió resultado, saltando época")
            return 0, 0, 0

        # 5. Promediar
        W_acc = [np.zeros_like(w) for w in W0]
        b_acc = [np.zeros_like(b) for b in b0]
        total_loss = total_acc = total_gnorm = 0.0

        for Wb, bb, loss, acc, gnorm in results:
            for i in range(len(W_acc)):
                W_acc[i] += Wb[i]
                b_acc[i] += bb[i]

            total_loss += loss
            total_acc += acc
            total_gnorm += gnorm

        m = len(results)
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / m
            layer.bias = b_acc[i] / m

        return total_loss / m, total_acc / m, total_gnorm / m
