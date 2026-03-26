from mnist.experiment import get_network, run_experiment
from nncore import Model
from nncore.costs import MeanSquaredError
from nncore.optimizers import SGD
from nncore.strategies import ParallelWeightAvgStrategy

from .mnist_data import MnistData


def build_model(lr=0.05):
    return Model(
        get_network(), MeanSquaredError(), SGD(learning_rate=lr), strategy=None
    )


def main():
    data = MnistData()
    data.download_data()
    data.load_data(one_hot=True)
    data.plot_random_samples(10)

    X_train, y_train, X_test, y_test = data.get_data()

    strategy = ParallelWeightAvgStrategy(6000, model_fn=build_model, shuffle=False)
    run_experiment(
        strategy,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
