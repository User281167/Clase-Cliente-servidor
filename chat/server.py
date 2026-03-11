import socket
import threading
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    CONNECT = "connect"  # Nueva conexión/registro
    RECONNECT = "reconnect"  # Reconexión con ID existente
    DISCONNECT = "disconnect"  # Desconexión intencional
    CHAT = "chat"  # Mensaje de chat normal
    PRIVATE = "private"  # Mensaje privado
    SYSTEM = "system"  # Mensaje del sistema
    PING = "ping"  # Keep-alive
    PONG = "pong"  # Respuesta keep-alive
    USER_LIST = "user_list"  # Lista de usuarios


@dataclass
class Message:
    type: str
    client_id: Optional[str] = None
    target_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[float] = None
    data: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str) -> "Message":
        return Message(**json.loads(data))


class ClientConnection:
    def __init__(self, socket: socket.socket, address: tuple):
        self.socket = socket
        self.address = address
        self.client_id: Optional[str] = None
        self.username: Optional[str] = None
        self.is_connected = True
        self.last_ping = time.time()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def send(self, message: Message) -> bool:
        try:
            with self.lock:
                if self.is_connected:
                    self.socket.send((message.to_json() + "\n").encode("utf-8"))
                    return True
        except Exception as e:
            print(f"Error enviando a {self.client_id}: {e}")
            self.is_connected = False
        return False

    def close(self):
        self.is_connected = False
        try:
            self.socket.close()
        except:
            pass


class TCPServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.clients: Dict[str, ClientConnection] = {}  # client_id -> ClientConnection
        self.pending_messages: Dict[str, List[Message]] = (
            {}
        )  # Mensajes pendientes para clientes desconectados
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        print(f"🚀 Servidor iniciado en {self.host}:{self.port}")

        # Hilo para monitorear conexiones (keep-alive)
        threading.Thread(target=self._monitor_connections, daemon=True).start()

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client = ClientConnection(client_socket, address)

                thread = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                client.thread = thread
                thread.start()

            except Exception as e:
                if self.running:
                    print(f"Error aceptando conexión: {e}")

    def _handle_client(self, client: ClientConnection):
        print(f"📥 Nueva conexión desde {client.address}")
        buffer = ""

        try:
            while client.is_connected and self.running:
                data = client.socket.recv(4096).decode("utf-8")
                if not data:
                    break

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    self._process_message(client, line.strip())

        except Exception as e:
            print(f"Error con cliente {client.client_id}: {e}")
        finally:
            self._handle_disconnect(client)

    def _process_message(self, client: ClientConnection, data: str):
        try:
            msg = Message.from_json(data)
            msg.timestamp = time.time()

            if msg.type == MessageType.CONNECT.value:
                self._handle_connect(client, msg)
            elif msg.type == MessageType.RECONNECT.value:
                self._handle_reconnect(client, msg)
            elif msg.type == MessageType.CHAT.value:
                self._handle_chat(client, msg)
            elif msg.type == MessageType.PRIVATE.value:
                self._handle_private(client, msg)
            elif msg.type == MessageType.PING.value:
                self._handle_ping(client)
            elif msg.type == MessageType.DISCONNECT.value:
                client.is_connected = False

        except json.JSONDecodeError:
            self._send_system(client, "Error: Formato JSON inválido")
        except Exception as e:
            print(f"Error procesando mensaje: {e}")

    def _handle_connect(self, client: ClientConnection, msg: Message):
        """Nuevo cliente"""
        client_id = msg.client_id or f"user_{id(client)}"
        username = msg.content or f"Usuario_{client_id[-6:]}"

        with self.lock:
            # Si ya existe, desconectar el anterior
            if client_id in self.clients:
                old_client = self.clients[client_id]
                old_client.send(
                    Message(
                        type=MessageType.SYSTEM.value,
                        content="Sesión iniciada en otro dispositivo",
                    )
                )
                old_client.close()

            client.client_id = client_id
            client.username = username
            self.clients[client_id] = client

        print(f"✅ Cliente conectado: {username} ({client_id})")

        # Enviar confirmación
        client.send(
            Message(
                type=MessageType.CONNECT.value,
                client_id=client_id,
                content=f"Bienvenido {username}!",
                data={"username": username},
            )
        )

        # Enviar mensajes pendientes si los hay
        self._send_pending_messages(client)

        # Notificar a otros
        self._broadcast(
            Message(
                type=MessageType.SYSTEM.value, content=f"{username} se ha unido al chat"
            ),
            exclude=client_id,
        )

        self._broadcast_user_list()

    def _handle_reconnect(self, client: ClientConnection, msg: Message):
        """Cliente existente reconectándose"""
        client_id = msg.client_id

        with self.lock:
            if client_id not in self.clients and client_id not in self.pending_messages:
                # ID no existe, tratar como nueva conexión
                self._handle_connect(client, msg)
                return

            # Recuperar info del cliente anterior si existe
            if client_id in self.clients:
                old_client = self.clients[client_id]
                username = old_client.username
                old_client.close()
            else:
                username = msg.content or client_id

            client.client_id = client_id
            client.username = username
            self.clients[client_id] = client

        print(f"🔄 Cliente reconectado: {username} ({client_id})")

        client.send(
            Message(
                type=MessageType.RECONNECT.value,
                client_id=client_id,
                content=f"Reconectado como {username}",
                data={"username": username, "reconnected": True},
            )
        )

        # Enviar mensajes pendientes
        self._send_pending_messages(client)

        self._broadcast_user_list()

    def _handle_chat(self, client: ClientConnection, msg: Message):
        """Mensaje de chat broadcast"""
        if not client.client_id:
            self._send_system(client, "Error: No autenticado")
            return

        broadcast_msg = Message(
            type=MessageType.CHAT.value,
            client_id=client.client_id,
            content=msg.content,
            timestamp=time.time(),
            data={"username": client.username},
        )

        self._broadcast(broadcast_msg)
        print(f"💬 [{client.username}]: {msg.content}")

    def _handle_private(self, client: ClientConnection, msg: Message):
        """Mensaje privado"""
        if not client.client_id or not msg.target_id:
            return

        private_msg = Message(
            type=MessageType.PRIVATE.value,
            client_id=client.client_id,
            target_id=msg.target_id,
            content=msg.content,
            timestamp=time.time(),
            data={"username": client.username},
        )

        # Enviar al destinatario
        sent = False
        with self.lock:
            if msg.target_id in self.clients:
                sent = self.clients[msg.target_id].send(private_msg)
            else:
                # Guardar como pendiente
                if msg.target_id not in self.pending_messages:
                    self.pending_messages[msg.target_id] = []
                self.pending_messages[msg.target_id].append(private_msg)
                sent = True

        # Confirmar al remitente
        if sent:
            client.send(
                Message(
                    type=MessageType.SYSTEM.value,
                    content=f"Mensaje privado enviado a {msg.target_id}",
                )
            )

    def _handle_ping(self, client: ClientConnection):
        client.last_ping = time.time()
        client.send(Message(type=MessageType.PONG.value))

    def _handle_disconnect(self, client: ClientConnection):
        """Manejar desconexión (no eliminar inmediatamente para permitir reconexión)"""
        if not client.client_id:
            client.close()
            return

        print(f"⚠️ Cliente desconectado: {client.username} ({client.client_id})")

        # No eliminar inmediatamente, mantener en "pending" por si reconecta
        # El hilo de monitoreo limpiará después de un timeout

        with self.lock:
            if (
                client.client_id in self.clients
                and self.clients[client.client_id] is client
            ):
                self.clients[client.client_id].is_connected = False

        client.close()

    def _send_pending_messages(self, client: ClientConnection):
        """Enviar mensajes acumulados mientras estaba desconectado"""
        with self.lock:
            if client.client_id in self.pending_messages:
                messages = self.pending_messages.pop(client.client_id, [])
                for msg in messages:
                    client.send(msg)
                if messages:
                    client.send(
                        Message(
                            type=MessageType.SYSTEM.value,
                            content=f"📨 {len(messages)} mensajes pendientes recuperados",
                        )
                    )

    def _broadcast(self, msg: Message, exclude: Optional[str] = None):
        """Enviar a todos los clientes conectados"""
        disconnected = []

        with self.lock:
            for cid, client in list(self.clients.items()):
                if cid != exclude and client.is_connected:
                    if not client.send(msg):
                        disconnected.append(cid)
                        # Guardar mensaje como pendiente
                        if cid not in self.pending_messages:
                            self.pending_messages[cid] = []
                        self.pending_messages[cid].append(msg)

        # Limpiar desconectados
        for cid in disconnected:
            if cid in self.clients and not self.clients[cid].is_connected:
                del self.clients[cid]

    def _broadcast_user_list(self):
        """Enviar lista actualizada de usuarios"""
        with self.lock:
            users = [
                {"id": c.client_id, "username": c.username, "online": c.is_connected}
                for c in self.clients.values()
            ]

        self._broadcast(
            Message(type=MessageType.USER_LIST.value, data={"users": users})
        )

    def _send_system(self, client: ClientConnection, message: str):
        client.send(Message(type=MessageType.SYSTEM.value, content=message))

    def _monitor_connections(self):
        """Monitorear conexiones inactivas y limpiar"""
        while self.running:
            time.sleep(5)
            now = time.time()
            to_remove = []

            with self.lock:
                for cid, client in list(self.clients.items()):
                    if not client.is_connected:
                        # Mantener por 60 segundos para permitir reconexión
                        if now - client.last_ping > 60:
                            to_remove.append(cid)
                    elif now - client.last_ping > 30:
                        # Enviar ping para verificar
                        client.send(Message(type=MessageType.PING.value))

            for cid in to_remove:
                with self.lock:
                    if cid in self.clients:
                        del self.clients[cid]
                print(f"🗑️ Cliente eliminado por timeout: {cid}")
                self._broadcast_user_list()

    def stop(self):
        self.running = False
        with self.lock:
            for client in self.clients.values():
                client.close()
        if self.server_socket:
            self.server_socket.close()


if __name__ == "__main__":
    server = TCPServer(port=5000)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servidor...")
        server.stop()
