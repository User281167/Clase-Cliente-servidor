import numpy as np

from nncore import History
from nncore.utils import get_batches

from .training_strategy import TrainingStrategy


class FinalAvgStrategy(TrainingStrategy):
    """
    Cada worker entrena local_epochs independientemente.
    El History refleja el progreso REAL del worker 0 época por época.
    Promedia pesos UNA sola vez al final.

    A diferencia de WeightAvgStrategy (que registra 1 punto por época global),
    esta estrategia expone el entrenamiento interno completo en el History.
    """

    def __init__(
        self,
        n_workers: int,
        local_epochs: int = 200,
        batch_size: int | None = None,
        shuffle=True,
    ):
        self.n_workers = n_workers
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def step(self, model, X, y) -> "History":
        # para el historial inyectar los valores de tests
        X_val = getattr(self, "_X_val", None)
        y_val = getattr(self, "_y_val", None)

        layers = model.network.trainable_layers()
        n = len(X)
        chunk_size = n // self.n_workers
        history = History()

        W0 = [l.weights.copy() for l in layers]
        b0 = [l.bias.copy() for l in layers]
        W_acc = [np.zeros_like(l.weights) for l in layers]
        b_acc = [np.zeros_like(l.bias) for l in layers]
        k = 0

        for X_k, y_k in get_batches(X, y, chunk_size, shuffle=self.shuffle):
            # Cada worker parte de W_0
            for i, layer in enumerate(layers):
                layer.weights = W0[i].copy()
                layer.bias = b0[i].copy()

            for ep in range(self.local_epochs):
                if self.batch_size is not None:
                    ep_gnorm = 0.0
                    n_b = 0

                    for X_b, y_b in get_batches(X_k, y_k, self.batch_size):
                        y_pred = model.network.forward(X_b, training=True)
                        delta = model.cost.derivative(y_b, y_pred)
                        model.network.backward(delta)
                        model.optimizer.step(layers)
                        ep_gnorm += model.compute_grad_norm()
                        n_b += 1

                    ep_gnorm /= n_b
                else:
                    y_pred = model.network.forward(X_k, training=True)
                    delta = model.cost.derivative(y_k, y_pred)
                    model.network.backward(delta)

                    model.optimizer.step(layers)
                    ep_gnorm = model.compute_grad_norm()

                # Solo worker 0 alimenta el History
                if k == 0:
                    y_ep = model.network.forward(X_k, training=False)
                    ep_loss = float(model.cost.function(y_k, y_ep))
                    ep_acc = float(model.accuracy(y_k, y_ep))

                    val_loss = val_acc = None
                    if X_val is not None and y_val is not None:
                        val_loss, val_acc = model.evaluate_split(X_val, y_val)

                    history.record(ep + 1, ep_loss, ep_acc, ep_gnorm, val_loss, val_acc)
            k += 1

            for i, layer in enumerate(layers):
                W_acc[i] += layer.weights
                b_acc[i] += layer.bias

        # Promedio único al final
        for i, layer in enumerate(layers):
            layer.weights = W_acc[i] / self.n_workers
            layer.bias = b_acc[i] / self.n_workers

        return history  # ← History completo
