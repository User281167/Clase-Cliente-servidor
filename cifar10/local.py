from nncore import Model, Network
from nncore.activations import LeakyReLU, Softmax
from nncore.costs import MeanSquaredError
from nncore.initializers import HeUniform
from nncore.layers import Dense, Dropout
from nncore.optimizers import SGD
from nncore.strategies import GradAvarageStrategy
from nncore.utils import print_system_info, time_wrapper

from .cifar10_data import Cifar10Data


@time_wrapper
def run(
    grayscale: bool = False,
    lr: float = 0.1,
    batch_size: int = 5000,
    epochs: int = 100,
):
    print_system_info()

    normal = HeUniform()

    net = Network()
    net.add(Dense(1024 if grayscale else 3072, 512, LeakyReLU(), normal))
    net.add(Dropout(p=0.2))
    net.add(Dense(512, 256, LeakyReLU(), normal))
    net.add(Dense(256, 128, LeakyReLU(), normal))
    net.add(Dropout(p=0.1))
    net.add(Dense(128, 10, Softmax(), normal))
    net.summary()

    strategy = GradAvarageStrategy(batch_size=batch_size)
    model = Model(net, MeanSquaredError(), SGD(learning_rate=lr), strategy=strategy)

    data = Cifar10Data()
    data.load_data(grayscale=grayscale)
    data.plot_random_samples()
    X_train, y_train, X_test, y_test = data.get_data()

    # shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
        verbose_epoch=epochs // 10,
    )

    history.plot()
    model.evaluate(X_test, y_test)
    model.confusion_matrix(X_test, y_test)
    model.classification_report(X_test, y_test)


if __name__ == "__main__":
    run()
