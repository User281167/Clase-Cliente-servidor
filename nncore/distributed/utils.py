import logging
import pickle
import socket

log = logging.getLogger("distributed_training")
log.setLevel(logging.DEBUG)

if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_h)

log.propagate = False

# ══════════════════════════════════════════════════════════════════════════════
# Helpers de red — framing simple: 4 bytes (big-endian) de longitud + payload
# ══════════════════════════════════════════════════════════════════════════════


def _pickle_code(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _send_raw(sock, raw: bytes):
    sock.sendall(len(raw).to_bytes(4, "big") + raw)


def _send_msg(sock, obj):
    raw = _pickle_code(obj)
    _send_raw(sock, raw)


def _send_safe(wid, sock, msg, is_raw=False):
    try:
        if is_raw:
            _send_raw(sock, msg)
        else:
            _send_msg(sock, msg)

        return None
    except Exception as e:
        log.warning(f"Worker {wid} error in send: {e}")
        return wid


def _recv_msg(sock):
    header = _recv_exact(sock, 4)
    length = int.from_bytes(header, "big")
    raw = _recv_exact(sock, length)
    return pickle.loads(raw)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()

    while len(buf) < n:
        chunk = sock.recv(n - len(buf))

        if not chunk:
            raise ConnectionError("Socket cerrado prematuramente")

        buf.extend(chunk)

    return bytes(buf)
