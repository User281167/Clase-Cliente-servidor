import matplotlib.pyplot as plt
import numpy as np

from .costs import CostFunction
from .history import History
from .network import Network
from .optimizers import Optimizer
from .strategies import FinalAvgStrategy, TrainingStrategy


class Model:
    """
    Orquesta el ciclo de entrenamiento delegando el paso
    por época a una TrainingStrategy intercambiable.

    Uso:
        model = Model(network, CrossEntropy(), Adam(), MiniBatchStrategy(64))
        history = model.fit(X_train, y_train, epochs=20)
    """

    def __init__(
        self,
        network: "Network",
        cost: "CostFunction",
        optimizer: "Optimizer",
        strategy: "TrainingStrategy",
    ):
        self.network = network
        self.cost = cost
        self.optimizer = optimizer
        self.strategy = strategy

    # ──────────────────────────────────────────
    # Entrenamiento
    # ──────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
        verbose_epoch: int = 1,
    ) -> "History":
        if isinstance(self.strategy, FinalAvgStrategy):
            self.strategy._X_val = X_val
            self.strategy._y_val = y_val

        history = History()

        for epoch in range(1, epochs + 1):
            result = self.strategy.step(self, X_train, y_train)

            # FinalAvgStrategy devuelve History directamente
            if isinstance(result, History):
                return result  # ya tiene todo registrado

            loss, acc, gnorm = result

            val_loss = val_acc = None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate_split(X_val, y_val)

            history.record(epoch, loss, acc, gnorm, val_loss, val_acc)

            if verbose and (
                epoch == 1 or epoch % verbose_epoch == 0 or epoch == epochs
            ):
                self._print_epoch(epoch, epochs, loss, acc, gnorm, val_loss, val_acc)

        return history

    def compute_grad_norm(self) -> float:
        """
        Norma de Frobenius de los gradientes de TODOS los pesos.

        ||∂L/∂W||_F = sqrt( Σ_layers Σ_ij (∂L/∂W_ij)² )

        Un valor único por paso que resume la magnitud global
        del gradiente en la red completa.

        Interpretación:
        < 0.001  → posible vanishing gradient
        > 100    → posible exploding gradient
        estable  → entrenamiento bien condicionado
        """
        total = 0.0

        for layer in self.network.trainable_layers():
            if layer.d_weights is not None:
                total += np.sum(layer.d_weights**2)
                total += np.sum(layer.d_bias**2)

        return float(np.sqrt(total))

    # ──────────────────────────────────────────
    # Evaluación y predicción
    # ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción en modo inference (Dropout desactivado).
        Retorna probabilidades shape (N, K).
        """
        return self.network.forward(X, training=False)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna la clase predicha por argmax.
        shape (N,) con valores en {0,...,K-1}
        """
        return np.argmax(self.predict(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """
        Evalúa la red sobre un conjunto completo.
        Retorna dict con loss y accuracy.
        """
        loss, acc = self.evaluate_split(X, y)
        if verbose:
            print(f"Loss: {loss:.4f}  |  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

        return {"loss": loss, "accuracy": acc}

    def evaluate_split(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.network.forward(X, training=False)
        loss = float(self.cost.function(y, y_pred))
        acc = float(self.accuracy(y, y_pred))

        return loss, acc

    def confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Matriz de confusión para clasificación multiclase.

        C[i, j] = número de muestras de clase i predichas como clase j

        Diagonal principal : predicciones correctas
        Fuera de diagonal  : errores — C[i,j] dice "clase i confundida con j"

        Para MNIST (10 clases) produce matriz 10×10.
        """
        y_pred = self.predict_classes(X)
        y_true = np.argmax(y, axis=1) if y.ndim == 2 else y.flatten().astype(int)
        K = len(np.unique(y_true))
        cm = np.zeros((K, K), dtype=int)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        if plot:
            self._plot_confusion_matrix(cm)

        return cm

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Visualiza la matriz de confusión con anotaciones.
        Normaliza por fila para mostrar % de acierto por clase.
        """
        K = cm.shape[0]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Conteos absolutos ──
        im0 = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Confusion Matrix — Conteos")
        plt.colorbar(im0, ax=axes[0])

        for i in range(K):
            for j in range(K):
                axes[0].text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )

        axes[0].set_xlabel("Predicho")
        axes[0].set_ylabel("Real")
        axes[0].set_xticks(range(K))
        axes[0].set_yticks(range(K))

        # ── Normalizada por fila (% accuracy por clase) ──
        im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Confusion Matrix — Normalizada")
        plt.colorbar(im1, ax=axes[1])

        for i in range(K):
            for j in range(K):
                axes[1].text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        axes[1].set_xlabel("Predicho")
        axes[1].set_ylabel("Real")
        axes[1].set_xticks(range(K))
        axes[1].set_yticks(range(K))

        plt.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Reporte por clase: precision, recall, F1-score.

        Precision_k = TP_k / (TP_k + FP_k)
        De todo lo que predije como k, ¿cuánto era realmente k?

        Recall_k = TP_k / (TP_k + FN_k)
        De todo lo que era k, ¿cuánto predije correctamente?

        F1_k = 2 · (Precision_k · Recall_k) / (Precision_k + Recall_k)
        Media armónica — balance entre precision y recall

        Útil para detectar qué dígitos confunde más la red.
        """
        cm = self.confusion_matrix(X, y, plot=False)
        K = cm.shape[0]
        eps = 1e-8

        print(f"\n{'─' * 52}")
        print(
            f"{'Clase':^8} {'Precision':^12} {'Recall':^12} {'F1':^12} {'Samples':^8}"
        )
        print(f"{'─' * 52}")

        precisions = recalls = f1s = 0.0

        for k in range(K):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            samples = cm[k, :].sum()

            precisions += precision
            recalls += recall
            f1s += f1

            print(f"{k:^8} {precision:^12.4f} {recall:^12.4f} {f1:^12.4f} {samples:^8}")

        print(f"{'─' * 52}")
        print(
            f"{'avg':^8} {precisions / K:^12.4f} {recalls / K:^12.4f} {f1s / K:^12.4f}"
        )
        print(f"{'─' * 52}\n")

    # ──────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Accuracy = (1/N) Σ_n 1[argmax(ŷ_n) == argmax(y_n)]

        Funciona con y_true one-hot o enteros.
        """
        pred = np.argmax(y_pred, axis=1)

        if y_true.ndim == 2:
            true = np.argmax(y_true, axis=1)
        else:
            true = y_true.flatten().astype(int)

        return np.mean(pred == true)

    # ──────────────────────────────────────────
    # Verbose
    # ──────────────────────────────────────────

    def _print_epoch(
        self,
        epoch: int,
        total: int,
        loss: float,
        acc: float,
        gnorm: float | None = None,
        val_loss=None,
        val_acc=None,
    ) -> None:
        bar_len = 20
        filled = int(bar_len * epoch / total)

        bar = "█" * filled + "░" * (bar_len - filled)
        line = (
            f"Epoch {epoch:>4}/{total} [{bar}] "
            f"loss={loss:.4f}  acc={acc:.4f}  "
            f"||g||={gnorm:.4f}"
        )

        if val_loss is not None:
            line += f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

        print(line)

    def __repr__(self) -> str:
        return (
            f"Model(\n"
            f"  network={repr(self.network)},\n"
            f"  cost={self.cost.__class__.__name__},\n"
            f"  optimizer={self.optimizer.__class__.__name__}\n"
            f")"
        )
