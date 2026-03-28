from nncore import Model, Network
from nncore.activations import LeakyReLU, ReLU, Softmax
from nncore.costs import CrossEntropy, MeanSquaredError
from nncore.initializers import HeUniform
from nncore.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from nncore.optimizers import SGD, Adam
from nncore.strategies import GradAvarageStrategy
from nncore.utils import print_system_info, time_wrapper

from .mnist_data import MnistData


@time_wrapper
def main():
    print_system_info()

    normal = HeUniform()

    net = Network()
    net.add(
        Conv2D(
            1,
            8,
            kernel_size=3,
            padding=1,
            activation=ReLU(),
            weight_initializer=normal,
        )
    )
    net.add(MaxPool2D(2))
    net.add(
        Conv2D(
            8,
            16,
            kernel_size=3,
            padding=1,
            activation=ReLU(),
            weight_initializer=normal,
        )
    )
    net.add(MaxPool2D(2))
    net.add(Flatten())
    net.add(Dropout(p=0.4))
    net.add(Dense(16 * 7 * 7, 10, Softmax(), normal))
    net.summary(input_shape=(1, 28, 28))

    strategy = GradAvarageStrategy(batch_size=256)
    model = Model(net, MeanSquaredError(), SGD(learning_rate=0.1), strategy=strategy)
    # model = Model(net, CrossEntropy(), Adam(), strategy=strategy)

    data = MnistData()
    data.download_data()
    data.load_data()

    X_train, y_train, X_test, y_test = data.get_data()
    X_train = X_train.reshape(-1, 1, 28, 28)  # (60000, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)  # (10000, 1, 28, 28)

    # shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
        verbose_epoch=1,
    )

    history.plot()
    model.evaluate(X_test, y_test)
    model.confusion_matrix(X_test, y_test)
    model.classification_report(X_test, y_test)


if __name__ == "__main__":
    main()
