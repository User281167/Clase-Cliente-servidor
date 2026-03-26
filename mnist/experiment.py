import numpy as np

from nncore import Model, Network
from nncore.activations import ReLU, Softmax
from nncore.costs import MeanSquaredError
from nncore.initializers import RandomNormal
from nncore.layers import Dense
from nncore.optimizers import SGD
from nncore.utils import print_system_info, time_wrapper

SEED = 42


def get_network():
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))
    return net


def build_model(lr=0.05):
    return Model(
        get_network(), MeanSquaredError(), SGD(learning_rate=lr), strategy=None
    )


@time_wrapper
def run_experiment(
    strategy,
    epochs=200,
    verbose_epoch=20,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
):
    print_system_info()

    net = get_network()
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
