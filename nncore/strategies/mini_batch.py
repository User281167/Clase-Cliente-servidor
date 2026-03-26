from nncore.utils import get_batches

from .training_strategy import TrainingStrategy


class MiniBatchStrategy(TrainingStrategy):
    """
    Mini-batch GD — optimizer.step() en CADA batch.

    Cada batch parte de los pesos actualizados por el anterior.

    época 1:
        batch 1 parte de W_0     → produce W_1
        batch 2 parte de W_1     → produce W_2  ← parte del ACTUALIZADO
        batch 3 parte de W_2     → produce W_3  ← parte del ACTUALIZADO

    Para cada batch b de tamaño B:
        ∂L/∂W_b = (1/B) Σ_{n∈b} ∂L_n/∂W
        W = W - η · ∂L/∂W_b

    Ventaja  : balance entre velocidad y estabilidad del gradiente
                introduce ruido que ayuda a escapar mínimos locales
    Desventaja: gradiente ruidoso — no es la dirección exacta de descenso

    batch_size típico: 32, 64, 128 para MNIST
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def step(self, model, X, y):
        epoch_loss = epoch_acc = epoch_gnorm = 0.0
        n_batches = 0

        for X_b, y_b in get_batches(X, y, self.batch_size):
            y_pred = model.network.forward(X_b, training=True)
            loss = model.cost.function(y_b, y_pred)
            delta = model.cost.derivative(y_b, y_pred)
            model.network.backward(delta)
            model.optimizer.step(model.network.trainable_layers())

            epoch_loss += float(loss)
            epoch_acc += float(model.accuracy(y_b, y_pred))
            epoch_gnorm += model.compute_grad_norm()
            n_batches += 1

        return epoch_loss / n_batches, epoch_acc / n_batches, epoch_gnorm / n_batches
