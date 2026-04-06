import argparse
import json
import os
from datetime import datetime

from nncore import Model, Network
from nncore.activations import LeakyReLU
from nncore.costs import CrossEntropy
from nncore.initializers import HeUniform
from nncore.layers import Dense, Dropout
from nncore.optimizers import Adam
from nncore.utils import print_system_info

from .cifar10_sampler import Cifar10DataSampler
from .client import DistributedGradientAvgStrategy
from .server import ServerGradientAvgStrategy


def get_model(lr: float, strategy=None, grayscale: bool = False) -> Model:
    normal = HeUniform()

    net = Network()
    net.add(Dense(1024 if grayscale else 3072, 128, LeakyReLU(), normal))
    net.add(Dropout(p=0.2))
    net.add(Dense(128, 32, LeakyReLU(), normal))
    net.add(Dense(32, 10, LeakyReLU(), normal))
    net.summary()

    return Model(net, CrossEntropy(), Adam(learning_rate=lr), strategy=strategy)


def run_server(
    server_host: str,
    server_port: int,
    workers: int = 1,
    min_workers: int = 1,
    lr=0.05,
    epochs=200,
    grayscale=True,
    normalize=True,
):
    train_dataset = Cifar10DataSampler(
        train=True, grayscale=grayscale, normalize=normalize
    )
    test_dataset = Cifar10DataSampler(
        train=False, grayscale=grayscale, normalize=normalize
    )

    strategy = ServerGradientAvgStrategy(
        batch_size=len(train_dataset) // workers, min_workers=min_workers
    )
    strategy.start_server(server_port, server_host)

    save_folder = f"metrics__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    X_test, y_test = test_dataset[:]

    try:
        model = get_model(lr, strategy)
        history = model.fit(
            X_train=None,  # server no tiene datos de entrenamiento usa sampler
            y_train=None,
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


def run_client(
    server_host: str, server_port: int, lr: float, grayscale: bool, normalize: bool
):
    train_dataset = Cifar10DataSampler(
        train=True, grayscale=grayscale, normalize=normalize
    )

    model = get_model(lr)
    strategy = DistributedGradientAvgStrategy(server_host, server_port)

    save_folder = f"metrics_client__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)

    try:
        strategy.connect()
        strategy.run(model, train_dataset)
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
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9999, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--min_workers", default=1, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.server:
        run_server(
            args.host,
            args.port,
            args.workers,
            args.min_workers,
            args.lr,
            args.epochs,
            args.grayscale,
            args.normalize,
        )
    elif args.client:
        run_client(args.host, args.port, args.lr, args.grayscale, args.normalize)
    else:
        raise ValueError("No se especificó --server o --client")

    return 0


if __name__ == "__main__":
    main()
