"""
═══════════════════════
Entrenamiento distribuido sobre TCP con la misma interfaz que las estrategias locales.

Arquitectura
────────────
                    ┌────────────────────────┐
                    │  ServerTrainingStrategy│  (ABC)  maneja TCP, serialización,
                    │                        │         broadcast, loop
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │ ServerWeightAvgStrategy│  implementa step() — lógica Batch-Avg
                    └────────────────────────┘

                    ┌──────────────────────┐
                    │    ClientStrategy    │  (ABC)  maneja conexión, loop de mensajes
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │DistributedBatchAvg   │  implementa run()  — ejecuta forward/back
                    │Strategy              │
                    └──────────────────────┘

Protocolo de mensajes (pickle sobre TCP)
───────────────────────────────────────────────
  SERVER → CLIENT
    {"type": "assign",   "worker_id": int, "seed": int, "start": int, "n_batches": int}
    {"type": "weights",  "payload": <pickle bytes b64>}   ← W0, b0
    {"type": "step"}                                      ← ejecuta un paso
    {"type": "done"}                                      ← cierra

  CLIENT → SERVER
    {"type": "ready",    "worker_id": int}
    {"type": "result",   "worker_id": int, "payload": <pickle bytes b64>}
                          payload = (Wb, bb, loss, acc, gnorm)

Uso — Servidor
──────────────
    strategy = ServerWeightAvgStrategy(batch_size=256, min_workers=2)
    strategy.start_server(port=9999)

    model = Model(net, MeanSquaredError(), SGD(0.05), strategy=strategy)
    history = model.fit(X_train, y_train, epochs=50)

Uso — Cliente
─────────────
    model = Model(net, MeanSquaredError(), SGD(0.05))   # optimizer ignorado
    strategy = DistributedWeightAvgStrategy("192.168.1.10", 9999)
    strategy.connect()
    strategy.run(model, X_train, y_train)               # blocking loop
"""

from __future__ import annotations

import selectors
import socket
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from nncore.distributed.utils import _pickle_code, _recv_msg, _send_msg, _send_safe, log

# ══════════════════════════════════════════════════════════════════════════════
# ABC — Servidor
# ══════════════════════════════════════════════════════════════════════════════


class ServerTrainingStrategy(ABC):
    """
    Base para estrategias de entrenamiento distribuido en el lado servidor.

    Gestiona:
      - TCP accept loop en hilo separado
      - Registro y reconexión de workers
      - Broadcast de pesos iniciales
      - Recolección de resultados
      - Serialización/deserialización (JSON + pickle b64)

    Subclases deben implementar `step()` con la lógica de agregación.
    """

    # ── Configuración ──────────────────────────────────────────────────────

    CONNECT_TIMEOUT = 120  # segundos esperando min_workers al inicio
    WORKER_TIMEOUT = 60  # segundos esperando resultado de un worker
    RECONNECT_WINDOW = 30  # segundos que un worker tiene para reconectarse
    PING_INTERVAL = 30  # segundos entre health-pings (heartbeat futuro)

    def __init__(self, min_workers: int = 1):
        self.min_workers = min_workers

        self._server_sock: Optional[socket.socket] = None
        self._port: Optional[int] = None

        # worker_id → socket activo
        self._workers: dict[int, socket.socket] = {}
        self._workers_lock = threading.Lock()

        # worker_id → metadatos asignados (seed, start, n_batches)
        self._assignments: dict[int, dict] = {}

        self._accept_thread: Optional[threading.Thread] = None
        self._running = False
        self._next_worker_id = 0

        # Se dispara cuando len(_workers) >= min_workers
        self._ready_event = threading.Event()

        # para broadcast
        self._pool = ThreadPoolExecutor(max_workers=32)

    # ── Ciclo de vida del servidor ─────────────────────────────────────────

    def start_server(self, port: int = 9999, host: str = "0.0.0.0") -> None:
        """Arranca el servidor TCP y espera min_workers conexiones."""
        self._port = port
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )  # Reusar dirección
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._server_sock.bind((host, port))
        self._server_sock.listen(32)
        self._running = True

        log.info(
            f"Servidor escuchando en {host}:{port} (min_workers={self.min_workers})"
        )

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # Espera min_workers workers listos — sin polling, Event se dispara
        # desde _handshake_worker en cuanto se alcanza el umbral
        log.info(f"Esperando {self.min_workers} worker(s)...")

        if not self._ready_event.wait(timeout=self.CONNECT_TIMEOUT):
            with self._workers_lock:
                n = len(self._workers)

            raise TimeoutError(
                f"Solo {n}/{self.min_workers} workers conectaron en {self.CONNECT_TIMEOUT}s"
            )

        with self._workers_lock:
            n = len(self._workers)

        log.info(f"{n} worker(s) conectados — servidor listo")

    def stop_server(self) -> None:
        """
        Cierre ordenado:
        1. Envía done a todos los workers y cierra sus sockets
        2. Detiene el accept loop (running=False)
        3. Cierra el server socket
        4. Apaga el thread pool
        El accept thread es daemon — no necesita join(), muere con el proceso
        o cuando el server socket se cierra y OSError interrumpe el accept().
        """
        if not self._running:
            return

        self._running = False

        # 1 — notificar y cerrar workers
        with self._workers_lock:
            for wid, sock in list(self._workers.items()):
                try:
                    _send_msg(sock, {"type": "done"})
                except Exception:
                    pass
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except Exception:
                    pass
            self._workers.clear()

        # 2 & 3 — cerrar server socket interrumpe accept() en el hilo daemon
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

            self._server_sock = None

        # 4 — pool
        self._pool.shutdown(wait=False)
        self._ready_event.clear()
        log.info("Servidor detenido")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop_server()

    # ── Accept loop (hilo daemon) ──────────────────────────────────────────

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)

        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handshake_worker,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handshake_worker(self, conn: socket.socket, addr) -> None:
        """Asigna worker_id y espera 'ready'."""
        try:
            msg = _recv_msg(conn)

            if msg.get("type") != "ready":
                conn.close()
                return

            # ¿reconexión?
            existing_id = msg.get("worker_id")

            with self._workers_lock:
                if existing_id is not None and existing_id in self._assignments:
                    wid = existing_id
                    log.info(f"Worker {wid} reconectado desde {addr}")
                else:
                    wid = self._next_worker_id
                    self._next_worker_id += 1
                    log.info(f"Nuevo worker {wid} desde {addr}")

                self._workers[wid] = conn

                # Disparar event si ya tenemos suficientes workers
                # Comenzar el entrenamiento
                if len(self._workers) >= self.min_workers:
                    self._ready_event.set()

            # Confirmar asignación
            assign = self._assignments.get(wid, {})
            _send_msg(conn, {"type": "assign", "worker_id": wid, **assign})

        except Exception as e:
            log.warning(f"Error en handshake con {addr}: {e}")
            conn.close()

    def _wait_workers(self) -> int | None:
        with self._workers_lock:
            n = len(self._workers)

        if n == 0:
            log.warning("No hay workers activos, esperando reconexion...")

            if not self._ready_event.wait(timeout=self.CONNECT_TIMEOUT):
                log.warning("Timeout esperando workers — saltando epoca")
                return None

            with self._workers_lock:
                n = len(self._workers)

        return n

    # ── Comunicación ───────────────────────────────────────────────────────
    def _broadcast_fast(self, msg):
        # No usa pool
        with self._workers_lock:
            workers = list(self._workers.items())

        raw = _pickle_code(msg)

        for wid, sock in workers:
            dead = _send_safe(wid, sock, raw, is_raw=True)
            if dead:
                self._remove_dead([dead])

    def _broadcast_pool(self, msg):
        with self._workers_lock:
            workers = list(self._workers.items())  # [(wid, sock), ...]

        if not workers:
            return

        raw = _pickle_code(msg)

        results = list(
            self._pool.map(
                lambda ws: _send_safe(ws[0], ws[1], raw, is_raw=True), workers
            )
        )
        dead = [wid for wid in results if wid is not None]

        if dead:
            self._remove_dead(dead)

    def _broadcast_weights(self, W0: list, b0: list) -> None:
        """Envía W0, b0 a todos los workers registrados."""
        msg = {"type": "weights", "payload": (W0, b0)}

        self._broadcast_pool(msg)

    def _broadcast_step(self) -> None:
        """Indica a todos los workers que ejecuten su paso."""
        self._broadcast_fast({"type": "step"})

    def _collect_results(self):
        sel = selectors.DefaultSelector()

        with self._workers_lock:
            items = list(self._workers.items())

        results = []
        dead = []

        for wid, sock in items:
            sel.register(sock, selectors.EVENT_READ, wid)

        deadline = time.time() + self.WORKER_TIMEOUT

        while sel.get_map():
            timeout = max(0, deadline - time.time())
            events = sel.select(timeout)

            if not events:
                break

            for key, _ in events:
                sock = key.fileobj
                wid = key.data

                sel.unregister(sock)

                try:
                    msg = _recv_msg(sock)

                    if msg["type"] != "result":
                        raise ValueError(msg["type"])

                    results.append(msg["payload"])
                except Exception as e:
                    log.warning(f"Worker {wid} falló: {e}")
                    dead.append(wid)

        for key in sel.get_map().values():
            dead.append(key.data)

        self._remove_dead(dead)

        return results

    def _remove_dead(self, wids: list) -> None:
        with self._workers_lock:
            for wid in wids:
                sock = self._workers.pop(wid, None)

                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

                log.warning(f"Worker {wid} removido")

            if not self._workers:
                log.warning("No quedan workers activos")
                self._ready_event.clear()

    # ── Abstracto ─────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, model, X: np.ndarray, y: np.ndarray):
        """Ejecuta una época completa y devuelve (loss, acc, gnorm)."""
        ...
