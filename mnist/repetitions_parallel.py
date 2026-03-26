from nncore.costs import MeanSquaredError
from nncore.optimizers import SGD
from nncore.strategies import ParallelWeightAvgStrategy
from nncore.utils import print_system_info

from .experiment import build_model, get_network
from .mnist_data import MnistData
from .study import run_study

REPETITIONS = 20
EPOCHS = 100
LR = 0.1
BATCH_SIZE = 6000
N_WORKERS = 10


def main():
    print_system_info()

    data = MnistData()
    data.download_data()
    data.load_data(one_hot=True)

    X_train, y_train, X_test, y_test = data.get_data()

    # ── WeightAvgStrategy ──
    study_parallel_batch_avg = run_study(
        strategy_factory=lambda: ParallelWeightAvgStrategy(
            BATCH_SIZE,
            model_fn=lambda: build_model(
                lr=LR
            ),  # ← captura LR para los procesos físicos
            shuffle=False,
        ),
        network_factory=get_network,
        cost=MeanSquaredError(),
        optimizer_factory=lambda: SGD(learning_rate=LR),
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        repetitions=REPETITIONS,
        epochs=EPOCHS,
        seeds=99,  # ← mismas seeds que los anteriores reportes sin ser paralelos
    )

    study_parallel_batch_avg.report()


if __name__ == "__main__":
    main()
