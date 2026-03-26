import argparse
import json
import os
from datetime import datetime

from nncore.activations import ReLU, Softmax
from nncore.costs import MeanSquaredError
from nncore.distributed import (
    DistributedGradientAvgStrategy,
    ServerGradientAvgStrategy,
)
from nncore.initializers import RandomNormal
from nncore.layers import Dense
from nncore.model import Model
from nncore.network import Network
from nncore.optimizers import SGD
from nncore.strategies import GradAvarageStrategy
from nncore.utils import print_system_info

from .mnist_data import MnistData


def get_net() -> Network:
    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))
    return net


def load_data() -> tuple:
    data = MnistData()
    data.download_data()
    data.load_data(one_hot=True)

    X_train, y_train, X_test, y_test = data.get_data()
    return X_train, y_train, X_test, y_test


def run_local(workers: int = 1, lr=0.05, epochs=200):
    X_train, y_train, X_test, y_test = load_data()

    net = get_net()
    net.summary()

    strategy = GradAvarageStrategy(batch_size=len(X_train) // workers)

    save_folder = f"local_metrics__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        model = Model(net, MeanSquaredError(), SGD(lr), strategy=strategy)
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose_epoch=25,
            X_val=X_test,
            y_val=y_test,
        )
        history.set_output_dir(save_folder)
        history.plot()
        model.evaluate(X_test, y_test)
        model.confusion_matrix(X_test, y_test)
        model.classification_report(X_test, y_test)
        history.save_all()
    except Exception as e:
        print(e)
    finally:
        strategy.metrics.to_csv(f"{save_folder}/metrics_server.csv", index=False)

        # Guardar descripción de métricas
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "epochs": epochs,
                    "workers": workers,
                },
                f,
            )


def run_server(
    server_host: str,
    server_port: int,
    workers: int = 1,
    min_workers: int = 1,
    lr=0.05,
    epochs=200,
):
    X_train, y_train, X_test, y_test = load_data()

    net = Network()
    net.add(Dense(784, 10, ReLU(), RandomNormal(0.01)))
    net.add(Dense(10, 10, Softmax(), RandomNormal(0.01)))

    strategy = ServerGradientAvgStrategy(
        batch_size=len(X_train) // workers, min_workers=min_workers
    )
    strategy.start_server(server_port, server_host)

    save_folder = f"metrics__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        model = Model(net, MeanSquaredError(), SGD(lr), strategy=strategy)
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            verbose_epoch=25,
            X_val=X_test,
            y_val=y_test,
        )
        history.set_output_dir(save_folder)
        history.plot()
        model.evaluate(X_test, y_test)
        model.confusion_matrix(X_test, y_test)
        model.classification_report(X_test, y_test)
        history.save_all()
    except Exception as e:
        print(e)
    finally:
        strategy.stop_server()
        strategy.metrics.to_csv(f"{save_folder}/metrics_server.csv", index=False)

        # Guardar descripción de métricas
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "epochs": epochs,
                    "workers": workers,
                    "min_workers": min_workers,
                },
                f,
            )


# %%
def run_client(server_host: str, server_port: int, lr=float):
    X_train, y_train, X_test, y_test = load_data()

    net = get_net()
    model = Model(net, MeanSquaredError(), SGD(lr))
    strategy = DistributedGradientAvgStrategy(server_host, server_port)

    save_folder = f"metrics_client__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        strategy.connect()
        strategy.run(model, X_train, y_train)
    except Exception as e:
        print(e)
        strategy.close()
    finally:
        # Guardar métricas locales del worker
        strategy.metrics.to_csv(f"{save_folder}/metrics_client.csv", index=False)

        # Guardar la descripción de las métricas (stats generales)
        description = strategy.metrics.describe(
            percentiles=[0.1, 0.5, 0.9], include="all"
        )
        description.to_csv(f"{save_folder}/metrics_description.csv", index=True)

        # Guardar args
        with open(f"{save_folder}/args.json", "w") as f:
            json.dump(
                {
                    "lr": lr,
                    "server_host": server_host,
                    "server_port": server_port,
                },
                f,
            )


def main():
    # arg 1 --server | --client
    # arg 2 --host
    # arg 3 --port
    #
    # ejemplo:
    # python mnist.py --server --host 0.0.0.0 --port 9999 --workers 2 --min_workers 1
    # python mnist.py --client --host 0.0.0.0 --port 9999
    #
    print_system_info()

    parser = argparse.ArgumentParser(description="Distributed Gradient Averaging")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9999, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--min_workers", default=1, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--epochs", default=200, type=int)

    args = parser.parse_args()
    print(args)

    if args.local:
        run_local(args.workers, args.lr, args.epochs)
    elif args.server:
        run_server(
            args.host, args.port, args.workers, args.min_workers, args.lr, args.epochs
        )
    elif args.client:
        run_client(args.host, args.port, args.lr)
    else:
        raise ValueError("No se especificó --local, --server o --client")

    return 0


if __name__ == "__main__":
    main()
