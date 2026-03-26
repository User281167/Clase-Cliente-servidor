from nncore.costs import MeanSquaredError
from nncore.optimizers import SGD
from nncore.strategies import FinalAvgStrategy, WeightAvgStrategy
from nncore.utils import print_system_info

from .experiment import get_network
from .mnist_data import MnistData
from .study import run_study

REPETITIONS = 20
EPOCHS = 100
LR = 0.1
BATCH_SIZE = 6000
N_WORKERS = 10
SEED_RNG = 99


def main():
    print_system_info()

    data = MnistData()
    data.download_data()
    data.load_data(one_hot=True)

    X_train, y_train, X_test, y_test = data.get_data()

    # ── WeightAvgStrategy ──
    study_batch_avg = run_study(
        strategy_factory=lambda: WeightAvgStrategy(batch_size=BATCH_SIZE),
        network_factory=get_network,
        cost=MeanSquaredError(),
        optimizer_factory=lambda: SGD(learning_rate=LR),
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        repetitions=REPETITIONS,
        epochs=EPOCHS,
        seeds=SEED_RNG,
    )

    # ── FinalAvgStrategy ──
    study_final_avg = run_study(
        strategy_factory=lambda: FinalAvgStrategy(
            n_workers=N_WORKERS, local_epochs=EPOCHS
        ),
        network_factory=get_network,
        cost=MeanSquaredError(),
        optimizer_factory=lambda: SGD(learning_rate=LR),
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        repetitions=REPETITIONS,
        epochs=1,  # FinalAvgStrategy maneja su propio loop
        seeds=SEED_RNG,
    )

    # ── Reportes ──
    study_batch_avg.report()
    study_final_avg.report()


if __name__ == "__main__":
    main()
