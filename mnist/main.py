from mnist.experiment import run_experiment
from nncore.strategies import WeightAvgStrategy

from .mnist_data import MnistData


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
