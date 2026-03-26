import os

import matplotlib.pyplot as plt
import pandas as pd


class History:
    """
    Registra métricas por epoch durante el entrenamiento.

    Atributos:
        train_loss     : list[float]  — loss promedio por epoch (train)
        train_accuracy : list[float]  — accuracy por epoch (train)
        train_grad_norm : norma de Frobenius promedio del gradiente por epoch
                          ||∂L/∂W||_F = sqrt(Σ_ij (∂L/∂W_ij)²)
                          Útil para detectar:
                            - grad_norm → 0   vanishing gradient
                            - grad_norm → ∞   exploding gradient
                            - grad_norm estable → entrenamiento saludable
        val_loss       : list[float]  — loss promedio por epoch (val)
        val_accuracy   : list[float]  — accuracy por epoch (val)
        epochs         : list[int]    — índices de epoch registrados
    """

    def __init__(self, output_dir: str | None = None):
        self.train_loss = []
        self.train_accuracy = []
        self.train_grad_norm = []
        self.val_loss = []
        self.val_accuracy = []
        self.epochs = []
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def record(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        grad_norm: float | None = None,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_acc)
        self.train_grad_norm.append(grad_norm if grad_norm is not None else 0.0)

        if val_loss is not None:
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_acc)

    def plot(self, show_grad_norm: bool = True) -> None:
        """
        Grafica las métricas registradas durante el entrenamiento.

        show_grad_norm=True agrega un tercer panel con la norma del gradiente.
        """
        has_val = len(self.val_loss) > 0
        has_grad = show_grad_norm and any(g > 0 for g in self.train_grad_norm)
        n_panels = 3 if has_grad else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

        # ── Loss ──
        axes[0].plot(self.epochs, self.train_loss, label="Train", color="steelblue")

        if has_val:
            axes[0].plot(
                self.epochs, self.val_loss, label="Val", color="tomato", linestyle="--"
            )

        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # ── Accuracy ──
        axes[1].plot(self.epochs, self.train_accuracy, label="Train", color="steelblue")

        if has_val:
            axes[1].plot(
                self.epochs,
                self.val_accuracy,
                label="Val",
                color="tomato",
                linestyle="--",
            )

        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # ── Grad norm ──
        if has_grad:
            axes[2].plot(
                self.epochs, self.train_grad_norm, color="seagreen", label="||∂L/∂W||_F"
            )
            axes[2].set_title("Gradient Norm")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("||∂L/∂W||_F")
            axes[2].legend()
            axes[2].grid(alpha=0.3)

        plt.suptitle("Training History", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "training_plot.png")
            plt.savefig(path, dpi=120)

        plt.show()

    def plot_comparison(
        self,
        other: "History",
        label_self: str = "Model A",
        label_other: str = "Model B",
    ) -> None:
        """
        Compara dos historiales — útil para comparar modos de GD,
        optimizadores, o arquitecturas distintas.

        Ejemplo:
            history_adam.plot_comparison(history_sgd,
                                         label_self="Adam",
                                         label_other="SGD")
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        pairs = [
            (self.train_loss, other.train_loss, "Loss", "Loss"),
            (self.train_accuracy, other.train_accuracy, "Accuracy", "Accuracy"),
            (
                self.train_grad_norm,
                other.train_grad_norm,
                "Gradient Norm",
                "||∂L/∂W||_F",
            ),
        ]
        colors = [
            ("steelblue", "tomato"),
            ("steelblue", "tomato"),
            ("seagreen", "darkorange"),
        ]

        for ax, (a, b, title, ylabel), (ca, cb) in zip(axes, pairs, colors):
            ax.plot(self.epochs, a, label=label_self, color=ca)
            ax.plot(other.epochs, b, label=label_other, color=cb, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def summary_table(self) -> None:
        """
        Imprime tabla resumen con best/last por métrica.

        ┌─────────────────────┬────────────┬────────────┐
        │ Métrica             │    Best    │    Last    │
        ├─────────────────────┼────────────┼────────────┤
        │ train_loss          │   0.0812   │   0.0934   │
        ...
        """
        metrics = [
            ("train_loss", self.train_loss, min),
            ("train_accuracy", self.train_accuracy, max),
            ("train_grad_norm", self.train_grad_norm, min),
        ]

        if self.val_loss:
            metrics += [
                ("val_loss", self.val_loss, min),
                ("val_accuracy", self.val_accuracy, max),
            ]

        sep = "─" * 45
        print(f"┌{sep}┐")
        print(f"│{'Training Summary':^45}│")
        print(f"├{'─' * 21}┬{'─' * 11}┬{'─' * 11}┤")
        print(f"│{'Métrica':<21}│{'Best':^11}│{'Last':^11}│")
        print(f"├{'─' * 21}┼{'─' * 11}┼{'─' * 11}┤")

        for name, values, best_fn in metrics:
            if values:
                best = best_fn(values)
                last = values[-1]
                print(f"│{name:<21}│{best:^11.4f}│{last:^11.4f}│")

        print(f"└{'─' * 21}┴{'─' * 11}┴{'─' * 11}┘")

    def save_csv(self, filename: str = "history.csv") -> None:
        data = {
            "epoch": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_accuracy,
            "grad_norm": self.train_grad_norm,
        }

        if self.val_loss:
            data["val_loss"] = self.val_loss
            data["val_acc"] = self.val_accuracy

        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)

    def save_plots(self):
        # Loss
        plt.figure()
        plt.plot(self.epochs, self.train_loss, label="Train")
        if self.val_loss:
            plt.plot(self.epochs, self.val_loss, label="Val")

        plt.legend()
        plt.title("Loss")
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close()

        # Accuracy
        plt.figure()
        plt.plot(self.epochs, self.train_accuracy, label="Train")
        if self.val_accuracy:
            plt.plot(self.epochs, self.val_accuracy, label="Val")

        plt.legend()
        plt.title("Accuracy")
        plt.savefig(os.path.join(self.output_dir, "accuracy.png"))
        plt.close()

        # Grad norm
        if any(g > 0 for g in self.train_grad_norm):
            plt.figure()
            plt.plot(self.epochs, self.train_grad_norm)
            plt.title("Gradient Norm")
            plt.savefig(os.path.join(self.output_dir, "grad_norm.png"))
            plt.close()

    def save_summary(self):
        path = os.path.join(self.output_dir, "summary.txt")

        with open(path, "w") as f:
            for name, values, fn in [
                ("train_loss", self.train_loss, min),
                ("train_acc", self.train_accuracy, max),
                ("grad_norm", self.train_grad_norm, min),
            ]:
                if values:
                    f.write(f"{name}: best={fn(values):.4f}, last={values[-1]:.4f}\n")

            if self.val_loss:
                f.write(
                    f"val_loss: best={min(self.val_loss):.4f}, last={self.val_loss[-1]:.4f}\n"
                )
                f.write(
                    f"val_acc: best={max(self.val_accuracy):.4f}, last={self.val_accuracy[-1]:.4f}\n"
                )

    def save_all(self):
        self.save_csv()
        self.save_plots()
        self.save_summary()

    def __repr__(self) -> str:
        if not self.epochs:
            return "History(vacío)"

        s = (
            f"History(epochs={self.epochs[-1]}, "
            f"train_loss={self.train_loss[-1]:.4f}, "
            f"train_acc={self.train_accuracy[-1]:.4f}, "
            f"grad_norm={self.train_grad_norm[-1]:.4f}"
        )

        if self.val_loss:
            s += (
                f", val_loss={self.val_loss[-1]:.4f}, "
                f"val_acc={self.val_accuracy[-1]:.4f}"
            )

        return s + ")"
