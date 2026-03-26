from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kagglehub import dataset_download


class MnistData:
    def __init__(self):
        self.dataset_path = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def download_data(self) -> None:
        self.dataset_path = dataset_download("oddrationale/mnist-in-csv")

    def load_data(self, one_hot: bool = True) -> None:
        train_df = pd.read_csv(self.dataset_path + "/mnist_train.csv")
        test_df = pd.read_csv(self.dataset_path + "/mnist_test.csv")

        self.X_train = train_df.iloc[
            :, 1:
        ].values  # 60000 imágenes de 784 píxeles (28x28)
        self.y_train = train_df.iloc[:, 0].values  # 60000 etiquetas (0-9)
        self.X_test = test_df.iloc[:, 1:].values
        self.y_test = test_df.iloc[:, 0].values

        self.X_train = self.X_train.astype("float32") / 255.0
        self.X_test = self.X_test.astype("float32") / 255.0

        if one_hot:
            self.y_train = self.one_hot_encode(self.y_train)
            self.y_test = self.one_hot_encode(self.y_test)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def plot_random_samples(self, n_samples=10) -> None:
        indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        samples = self.X_train[indices]
        labels = self.y_train[indices]

        # labels a entero si es one-hot
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)

        plt.figure(figsize=(10, 1))

        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap="gray")
            plt.title(f"Label: {labels[i]}")
            plt.axis("off")

        plt.show()

    def train_val_split(self, val_ratio: float = 0.1, seed: int | None = None):
        """
        Particiona X_train/y_train en train+val.
        val_ratio: fracción para validación, ej. 0.1 → 6000 muestras
        """
        np.random.seed(seed)
        n = len(self.X_train)
        np.random.seed()
        idx = np.random.permutation(n)
        split = int(n * val_ratio)
        val_idx, train_idx = idx[:split], idx[split:]

        return (
            self.X_train[train_idx],
            self.y_train[train_idx],
            self.X_train[val_idx],
            self.y_train[val_idx],
        )

    def one_hot_encode(self, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
        """
        Convierte etiquetas enteras a vectores one-hot.

        y: array (N,) con valores en {0,...,9}
        retorna: (N, 10) donde cada fila es e_k
        """
        one_hot = np.zeros((len(y), n_classes), dtype="float32")
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot
