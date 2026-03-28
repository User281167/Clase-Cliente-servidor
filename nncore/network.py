import numpy as np
from rich.console import Console
from rich.table import Table


# ──────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────
#
#  Representa el grafo de cómputo de la red neuronal.
#  Conecta capas secuencialmente y gestiona el flujo
#  de datos en ambas direcciones.
#
#  FORWARD PASS
#  ────────────
#  Dada entrada X de shape (N, n_in), propaga por cada capa:
#
#    A_0 = X
#    A_l = f_l(A_{l-1})    l = 1, ..., L
#
#  La salida A_L es la predicción de la red.
#
#  BACKWARD PASS
#  ─────────────
#  Recibe el gradiente inicial δ = ∂L/∂A_L de la loss,
#  propaga en orden inverso:
#
#    δ_{l-1} = layer_l.backward(δ_l)    l = L, ..., 1
#
#  Cada capa acumula d_weights y d_bias internamente.
#  El optimizer los lee después con optimizer.step(layers).
#
#  TRAINING vs INFERENCE
#  ──────────────────────
#  Dropout se comporta distinto en cada modo.
#  Network gestiona el flag training en todas las capas
#  antes de cada forward — el usuario no tiene que hacerlo.
#
class Network:
    """
    Grafo de cómputo secuencial.

    No conoce la loss, el optimizer ni los datos —
    solo sabe propagar hacia adelante y hacia atrás.

    Uso básico:
        net = Network()
        net.add(Dense(784, 128, Tanh(), XavierUniform()))
        net.add(Dropout(0.3))
        net.add(Dense(128, 10, Softmax(), XavierUniform()))

        y_pred = net.forward(X)          # inference por default
        y_pred = net.forward(X, training=True)  # training
        grad   = net.backward(delta)
    """

    def __init__(self):
        super().__init__()
        self.layers = []

    # ──────────────────────────────────────────
    # Construcción
    # ──────────────────────────────────────────

    def add(self, layer) -> "Network":
        """
        Agrega una capa al final de la red.
        Retorna self para permitir encadenamiento:

            net.add(Dense(...)).add(Dropout(...)).add(Dense(...))
        """
        self.layers.append(layer)
        return self

    def __len__(self) -> int:
        return len(self.layers)

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Propaga X por todas las capas en orden.

        training : True  durante entrenamiento (activa Dropout, etc.)
                   False durante inference (default)

        A_0 = X
        A_l = layer_l(A_{l-1})    l = 1,...,L

        Retorna A_L de shape (N, n_out)
        """
        self._set_training(training)

        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def __call__(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(X, training)

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Propaga el gradiente en orden inverso.

        delta : ∂L/∂A_L — gradiente de la loss respecto
                a la salida de la última capa. Shape (N, n_out).

        Cada capa acumula d_weights y d_bias.
        Retorna el gradiente respecto a la entrada X (raramente
        necesario fuera de redes más complejas).

        Nota:
            Llamar siempre DESPUÉS de forward(training=True).
            backward sin forward previo usa cachés desactualizados.
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def reset_weights(self) -> None:
        """
        Reinicializa W y b de todas las capas entrenables.
        Llamar entre experimentos para comparación justa.
        """
        for layer in self.trainable_layers():
            layer.weights = layer.initializer.initialize(
                layer.input_size, layer.output_size
            )
            layer.bias = np.zeros((1, layer.output_size))

    # ──────────────────────────────────────────
    # Parámetros
    # ──────────────────────────────────────────

    def trainable_layers(self) -> list:
        """
        Retorna solo las capas con parámetros aprendibles (W, b).
        El optimizer llama esto para saber qué actualizar.
        """
        return [l for l in self.layers if l.weights is not None]

    def parameter_count(self) -> int:
        """
        Cuenta el total de parámetros aprendibles de la red.

        Para una capa Dense(n_in, n_out):
          params = n_in * n_out  (W)  +  n_out  (b)
        """
        total = 0
        for layer in self.trainable_layers():
            total += layer.weights.size + layer.bias.size
        return total

    # ──────────────────────────────────────────
    # Persistencia  —  guardar y cargar pesos
    # ──────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        """
        Guarda W y b de cada capa entrenable en un archivo .npz.

        Formato: weights_0, bias_0, weights_1, bias_1, ...
        El índice corresponde a la posición en trainable_layers().
        """
        data = {}
        for i, layer in enumerate(self.trainable_layers()):
            data[f"weights_{i}"] = layer.weights
            data[f"bias_{i}"] = layer.bias

        np.savez(path, **data)

    def load_weights(self, path: str) -> None:
        """
        Carga pesos desde un .npz guardado con save_weights().

        Nota:
            La arquitectura debe ser idéntica a la que generó el archivo.
        """
        data = np.load(path)
        layers = self.trainable_layers()

        for i, layer in enumerate(layers):
            key_w = f"weights_{i}"
            key_b = f"bias_{i}"

            if key_w not in data or key_b not in data:
                raise ValueError(
                    f"Archivo no contiene pesos para capa {i}. "
                    f"¿La arquitectura coincide con el archivo?"
                )

            if data[key_w].shape != layer.weights.shape:
                raise ValueError(
                    f"Shape mismatch en capa {i}: "
                    f"archivo={data[key_w].shape}, "
                    f"red={layer.weights.shape}"
                )

            layer.weights = data[key_w]
            layer.bias = data[key_b]

    # ──────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────

    def _set_training(self, training: bool) -> None:
        """
        Propaga el flag training a todas las capas que lo usen (Dropout).
        Se llama automáticamente en forward().
        """
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training

    def summary(self, input_shape=None) -> None:
        """
        Imprime la arquitectura de la red con shapes y parámetros.

        Ejemplo:
        ┌─────────────────────────────────────────────────┐
        │                  Network Summary                │
        ├──────────────────────┬───────────┬──────────────┤
        │ Layer                │ Shape     │ Params       │
        ├──────────────────────┼───────────┼──────────────┤
        │ Dense (Tanh)         │ 784→128   │ 100,480      │
        │ Dropout (p=0.3)      │    —      │       0      │
        │ Dense (Softmax)      │ 128→10    │   1,290      │
        ├──────────────────────┼───────────┼──────────────┤
        │ Total params         │           │ 101,770      │
        └─────────────────────────────────────────────────┘
        """
        console = Console()
        table = Table(title="Network Summary", show_lines=True)

        table.add_column("Layer", style="cyan")
        table.add_column("Shape", justify="center", style="magenta")
        table.add_column("Params", justify="right", style="green")

        total = 0
        shape = input_shape

        for layer in self.layers:
            if hasattr(layer, "activation") and layer.activation is not None:
                name = f"{layer.__class__.__name__} ({layer.activation.__class__.__name__})"
            else:
                name = layer.__class__.__name__

            new_shape = layer.compute_output_shape(shape)
            shape_str = f"{shape} → {new_shape}"
            shape = new_shape

            if getattr(layer, "weights", None) is not None:
                params = layer.weights.size + layer.bias.size
                total += params
            else:
                params = 0

            table.add_row(name, shape_str, f"{params:,}")

        table.add_section()
        table.add_row("Total params", "", f"[bold]{total:,}[/bold]")

        console.print(table)

    def __repr__(self) -> str:
        lines = [f"Network({len(self.layers)} layers)"]

        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {repr(layer)}")

        return "\n".join(lines)
