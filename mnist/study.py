from dataclasses import dataclass, field

import numpy as np

from nncore import Model
from nncore.utils import time_wrapper

# ──────────────────────────────────────────────
# Resultado de una repetición
# ──────────────────────────────────────────────


@dataclass
class TrialResult:
    seed: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


# ──────────────────────────────────────────────
# Resultado agregado del estudio
# ──────────────────────────────────────────────


@dataclass
class StudyResult:
    strategy_name: str
    trials: list[TrialResult] = field(default_factory=list)

    # ── Estadísticas ──────────────────────────

    def _stats(self, values: list[float]) -> dict:
        a = np.array(values)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    @property
    def train_acc_stats(self):
        return self._stats([t.train_acc for t in self.trials])

    @property
    def val_acc_stats(self):
        return self._stats([t.val_acc for t in self.trials])

    @property
    def train_loss_stats(self):
        return self._stats([t.train_loss for t in self.trials])

    @property
    def val_loss_stats(self):
        return self._stats([t.val_loss for t in self.trials])

    # ── Print ─────────────────────────────────

    def _print_stats(self, label: str, stats: dict) -> None:
        print(f"\n--- {self.strategy_name} | {label} ---")
        print(f"  Media:          {stats['mean']:.4f}")
        print(f"  Desv. Estándar: {stats['std']:.4f}")
        print(f"  Mínimo:         {stats['min']:.4f}")
        print(f"  Máximo:         {stats['max']:.4f}")
        print(f"  Mediana:        {stats['median']:.4f}")

    def report(self) -> None:
        self._print_stats("Accuracy  TEST", self.val_acc_stats)
        self._print_stats("Accuracy  TRAIN", self.train_acc_stats)
        self._print_stats("Loss      TEST", self.val_loss_stats)
        self._print_stats("Loss      TRAIN", self.train_loss_stats)


# ──────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────
@time_wrapper
def run_study(
    strategy_factory,  # callable que devuelve una estrategia nueva: lambda: WeightAvgStrategy(6000)
    network_factory,  # callable que devuelve una red nueva:         get_network
    cost,
    optimizer_factory,  # callable que devuelve un optimizer nuevo:    lambda: SGD(0.05)
    X_train,
    y_train,
    X_val,
    y_val,
    repetitions: int = 30,
    epochs: int = 200,
    seeds: list[int] | int | None = None,
    verbose: bool = True,
) -> StudyResult:
    """
    Repite el entrenamiento `repetitions` veces con seeds distintas.
    Cada repetición usa una instancia nueva de red, optimizer y estrategia.
    No modifica ninguna clase existente.

    Parámetros:
        strategy_factory  : lambda sin args → TrainingStrategy
        network_factory   : lambda sin args → Network
        optimizer_factory : lambda sin args → Optimizer
        seeds             : lista de ints; si None se generan automáticamente; si int se generan automáticamente reproducibilidad

    Retorna:
        StudyResult con estadísticas agregadas
    """
    if seeds is None or isinstance(seeds, int):
        rng = np.random.default_rng(seeds)
        seeds = rng.integers(0, 100_000, size=repetitions).tolist()

    strategy_name = strategy_factory().__class__.__name__
    result = StudyResult(strategy_name=strategy_name)

    for i, seed in enumerate(seeds):
        if verbose:
            print(
                f"[{strategy_name}] rep {i + 1:>3}/{repetitions}  seed={seed}", end="\r"
            )

        # ── Inicialización reproducible ──
        np.random.seed(seed)
        net = network_factory()
        np.random.seed()  # liberar seed para training aleatorio

        strategy = strategy_factory()
        optimizer = optimizer_factory()
        model = Model(net, cost, optimizer, strategy=strategy)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            X_val=X_val,
            y_val=y_val,
            verbose=False,  # silenciar epochs individuales
        )

        # ── Métricas finales de la repetición ──
        train_loss = history.train_loss[-1]
        train_acc = history.train_accuracy[-1]
        val_loss = history.val_loss[-1]
        val_acc = history.val_accuracy[-1]

        result.trials.append(
            TrialResult(
                seed=seed,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )

    if verbose:
        print()  # salto de línea tras el \r

    return result
