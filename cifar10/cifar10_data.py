import tempfile
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

cifar10_classes = (
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
)


class Cifar10Data:
    def __init__(self, dataset_path: str | None = None):
        self.dataset_path = (
            dataset_path if dataset_path is not None else tempfile.gettempdir()
        )
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, grayscale=False, flatten=True, one_hot=True, normalize=False):
        """
        Args:
            grayscale (bool): convertir a 1 canal
            flatten (bool): flatten image to vector
            one_hot (bool): one-hot encode labels
            normalize (bool): normalize image a [-1, 1]
        Returns:
            X_train, y_train, X_test, y_test
        """

        transform_list = []

        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        transform_list.append(transforms.ToTensor())

        if normalize:
            mean = (0.5,) if grayscale else (0.5, 0.5, 0.5)
            std = (0.5,) if grayscale else (0.5, 0.5, 0.5)
            transform_list.append(transforms.Normalize(mean, std))

        if flatten:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

        transform = transforms.Compose(transform_list)

        trainset = torchvision.datasets.CIFAR10(
            root=self.dataset_path, train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=self.dataset_path, train=False, download=True, transform=transform
        )

        self.X_train = np.stack([img.numpy() for img, _ in trainset])
        self.y_train = np.array([label for _, label in trainset])

        self.X_test = np.stack([img.numpy() for img, _ in testset])
        self.y_test = np.array([label for _, label in testset])

        if one_hot:
            self.y_train = self.one_hot_encode(self.y_train)
            self.y_test = self.one_hot_encode(self.y_test)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def one_hot_encode(self, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
        one_hot = np.zeros((len(y), n_classes), dtype="float32")
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot

    def plot_random_samples(self, n_samples: int = 10, train: bool = True) -> None:
        X = self.X_train if train else self.X_test
        y = self.y_train if train else self.y_test

        if X is None:
            raise ValueError("Data not loaded yet.")

        indices = np.random.choice(len(X), n_samples, replace=False)
        samples = X[indices]
        labels = y[indices]

        # Convert one-hot to integers if needed
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)

        plt.figure(figsize=(10, 1))

        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)

            if samples.shape[1] == 1024:  # grayscale
                plt.imshow(samples[i].reshape(32, 32), cmap="gray")
            else:
                plt.imshow(samples[i].reshape(3, 32, 32).transpose(1, 2, 0))

            plt.title(f"{cifar10_classes[labels[i]]}")
            plt.axis("off")

        plt.show()
