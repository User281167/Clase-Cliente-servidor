import numpy as np

from nncore import Model, Network
from nncore.activations import ReLU, Softmax
from nncore.costs import MeanSquaredError
from nncore.initializers import RandomNormal
from nncore.layers import Dense
from nncore.optimizers import SGD
from nncore.strategies import WeightAvgStrategy
from nncore.utils import print_system_info, time_wrapper

from .mnist_data import MnistData

SEED = 42


@time_wrapper
def run_experiment(
    strategy,
    epochs=2,
    verbose_epoch=20,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):
    print_system_info()

    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))
    net.summary()

    np.random.seed(SEED)
    net.reset_weights()
    np.random.seed()

    model = Model(net, MeanSquaredError(), SGD(learning_rate=0.05), strategy=strategy)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
        verbose_epoch=verbose_epoch,
    )

    history.plot()
    model.evaluate(X_test, y_test)
    model.confusion_matrix(X_test, y_test)
    model.classification_report(X_test, y_test)

    return history


def main():
    data = MnistData()
    data.download_data()
    data.load_data(one_hot=True)
    data.plot_random_samples(10)

    X_train, y_train, X_test, y_test = data.get_data()

    run_experiment(
        strategy=WeightAvgStrategy(batch_size=6000),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
