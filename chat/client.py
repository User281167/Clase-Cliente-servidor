import socket
import threading
import json
import time
import uuid
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    CONNECT = "connect"
    RECONNECT = "reconnect"
    DISCONNECT = "disconnect"
    CHAT = "chat"
    PRIVATE = "private"
    SYSTEM = "system"
    PING = "ping"
    PONG = "pong"
    USER_LIST = "user_list"


@dataclass
class Message:
    type: str
    client_id: Optional[str] = None
    target_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[float] = None
    data: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "client_id": self.client_id,
                "target_id": self.target_id,
                "content": self.content,
                "timestamp": self.timestamp,
                "data": self.data,
            }
        )

    @staticmethod
    def from_json(data: str) -> "Message":
        return Message(**json.loads(data))


class TCPClient:
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.client_id: Optional[str] = None
        self.username: Optional[str] = None
        self.connected = False
        self.reconnecting = False
        self.should_run = True
        self.receive_thread: Optional[threading.Thread] = None
        self.ping_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_message: Optional[Callable[[Message], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_reconnect: Optional[Callable[[], None]] = None
        self.on_user_list: Optional[Callable[[list], None]] = None

        # Generar ID único persistente (en app real, guardar en archivo/config)
        self._persistent_id = self._load_client_id()

    def _load_client_id(self) -> str:
        """Cargar ID guardado o generar nuevo"""
        try:
            with open(".client_id", "r") as f:
                return f.read().strip()
        except:
            new_id = str(uuid.uuid4())[:8]
            self._save_client_id(new_id)
            return new_id

    def _save_client_id(self, client_id: str):
        """Guardar ID para futuras sesiones"""
        try:
            with open(".client_id", "w") as f:
                f.write(client_id)
        except:
            pass

    def connect(self, username: Optional[str] = None, reconnect: bool = False) -> bool:
        """Conectar o reconectar al servidor"""
        self.username = username or f"User_{self._persistent_id[:4]}"

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)

            if reconnect and self.client_id:
                # Intentar reconexión con ID anterior
                self._send(
                    Message(
                        type=MessageType.RECONNECT.value,
                        client_id=self.client_id,
                        content=self.username,
                    )
                )
            else:
                # Nueva conexión
                self._send(
                    Message(
                        type=MessageType.CONNECT.value,
                        client_id=self._persistent_id,
                        content=self.username,
                    )
                )

            self.connected = True
            self.reconnecting = False

            # Iniciar hilos
            self.receive_thread = threading.Thread(
                target=self._receive_loop, daemon=True
            )
            self.receive_thread.start()

            self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
            self.ping_thread.start()

            return True

        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            self.connected = False
            return False

    def _receive_loop(self):
        """Bucle principal de recepción"""
        buffer = ""

        while self.should_run and self.connected:
            try:
                data = self.socket.recv(4096).decode("utf-8")
                if not data:
                    raise ConnectionError("Servidor cerró conexión")

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    self._process_message(line.strip())

            except Exception as e:
                if self.should_run:
                    print(f"⚠️ Desconectado: {e}")
                    self._handle_disconnect()
                break

    def _process_message(self, data: str):
        try:
            msg = Message.from_json(data)

            if msg.type == MessageType.CONNECT.value:
                self.client_id = msg.client_id
                self._persistent_id = msg.client_id
                self._save_client_id(msg.client_id)
                print(f"✅ Conectado como {msg.data.get('username', 'Unknown')}")
                if self.on_connect:
                    self.on_connect()

            elif msg.type == MessageType.RECONNECT.value:
                print(f"🔄 Reconectado exitosamente")
                if self.on_reconnect:
                    self.on_reconnect()

            elif msg.type == MessageType.PONG.value:
                pass  # Keep-alive recibido

            elif msg.type == MessageType.USER_LIST.value:
                if self.on_user_list:
                    self.on_user_list(msg.data.get("users", []))

            elif msg.type == MessageType.SYSTEM.value:
                print(f"🔔 Sistema: {msg.content}")

            elif msg.type in [MessageType.CHAT.value, MessageType.PRIVATE.value]:
                sender = (
                    msg.data.get("username", msg.client_id)
                    if msg.data
                    else msg.client_id
                )
                msg_type = (
                    "PRIVADO" if msg.type == MessageType.PRIVATE.value else "CHAT"
                )
                print(f"[{msg_type}] {sender}: {msg.content}")

            if self.on_message:
                self.on_message(msg)

        except Exception as e:
            print(f"Error procesando mensaje: {e}")

    def _ping_loop(self):
        """Enviar keep-alive periódico"""
        while self.should_run and self.connected:
            time.sleep(15)
            if self.connected:
                self._send(Message(type=MessageType.PING.value))

    def _send(self, msg: Message) -> bool:
        try:
            self.socket.send((msg.to_json() + "\n").encode("utf-8"))
            return True
        except Exception as e:
            print(f"Error enviando: {e}")
            self._handle_disconnect()
            return False

    def send_chat(self, message: str) -> bool:
        """Enviar mensaje al chat general"""
        if not self.connected:
            print("❌ No conectado")
            return False
        return self._send(
            Message(
                type=MessageType.CHAT.value, client_id=self.client_id, content=message
            )
        )

    def send_private(self, target_id: str, message: str) -> bool:
        """Enviar mensaje privado"""
        if not self.connected:
            print("❌ No conectado")
            return False
        return self._send(
            Message(
                type=MessageType.PRIVATE.value,
                client_id=self.client_id,
                target_id=target_id,
                content=message,
            )
        )

    def _handle_disconnect(self):
        """Manejar desconexión inesperada"""
        self.connected = False
        try:
            self.socket.close()
        except:
            pass

        if self.on_disconnect:
            self.on_disconnect()

        # Auto-reconexión
        if self.should_run and not self.reconnecting:
            self._attempt_reconnect()

    def _attempt_reconnect(self):
        """Intentar reconectar automáticamente"""
        self.reconnecting = True
        attempts = 0
        max_attempts = 10

        while self.should_run and not self.connected and attempts < max_attempts:
            attempts += 1
            print(f"🔄 Intentando reconectar... ({attempts}/{max_attempts})")

            if self.connect(self.username, reconnect=True):
                print("✅ Reconexión exitosa")
                return True

            time.sleep(min(2**attempts, 30))  # Backoff exponencial

        if not self.connected:
            print("❌ No se pudo reconectar")
        self.reconnecting = False
        return False

    def disconnect(self):
        """Desconexión intencional"""
        self.should_run = False
        self.connected = False

        if self.socket:
            try:
                self._send(Message(type=MessageType.DISCONNECT.value))
                self.socket.close()
            except:
                pass

        print("👋 Desconectado del servidor")


def main():
    client = TCPClient(host="localhost", port=5000)

    # Configurar callbacks opcionales
    client.on_connect = lambda: print("🎉 Callback: Conectado!")
    client.on_reconnect = lambda: print("🎉 Callback: Reconectado!")
    client.on_disconnect = lambda: print("😢 Callback: Desconectado!")

    # Conectar
    username = input("Tu nombre de usuario: ").strip() or None
    if not client.connect(username):
        return

    print("\nComandos:")
    print("  /msg <id> <mensaje>  - Mensaje privado")
    print("  /usuarios            - Lista de usuarios (recibida automáticamente)")
    print("  /salir               - Desconectar")
    print("  <mensaje>            - Mensaje al chat general")
    print()

    try:
        while client.should_run:
            msg = input()

            if msg.startswith("/msg "):
                parts = msg[5:].split(" ", 1)
                if len(parts) == 2:
                    client.send_private(parts[0], parts[1])
                else:
                    print("Uso: /msg <id> <mensaje>")

            elif msg == "/usuarios":
                print("La lista se actualiza automáticamente...")

            elif msg == "/salir":
                break

            elif msg.strip():
                client.send_chat(msg)

    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
