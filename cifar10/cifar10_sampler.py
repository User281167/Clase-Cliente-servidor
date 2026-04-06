import tempfile

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Cifar10DataSampler:
    """
    Dataset sampler para CIFAR-10.

    Cargar los datos en RAM utilizando torchvision.
    """

    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        grayscale: bool = True,
        normalize: bool = True,
        flatten: bool = True,
        one_hot: bool = True,
        n_classes: int = 10,
    ):
        """
        Inicializa el dataset sampler para CIFAR-10.

        Args:
            root (str | None): Directorio donde se guardarán los datos. Si es None, se usa un directorio temporal.
            train (bool): Si es True, carga el conjunto de entrenamiento; si es False, carga el conjunto de prueba.
            normalize (bool): Si es True, normaliza los datos valores entre -1 y 1, False valores entre 0 y 1.
        """

        self.grayscale = grayscale
        self.flatten = flatten
        self.one_hot = one_hot
        self.n_classes = n_classes

        if root is None:
            root = tempfile.gettempdir()

        transform_list = []

        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        transform_list.append(transforms.ToTensor())  # valores [0,1] float32

        if normalize:
            # valores de -1 a 1
            mean = (0.5,) if grayscale else (0.5, 0.5, 0.5)
            std = (0.5,) if grayscale else (0.5, 0.5, 0.5)
            transform_list.append(transforms.Normalize(mean=mean, std=std))

        transform = transforms.Compose(transform_list)

        dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

        # cargar en RAM
        loader = DataLoader(
            dataset,
            batch_size=len(dataset),  # todo el dataset
            num_workers=0,
            shuffle=False,
        )
        images, labels = next(iter(loader))  # Tensor [N, C, H, W], Tensor [N]

        # Convertir a NumPy
        self._x = images.numpy().astype(np.float32)  # (N, C, H, W)
        self._y = labels.numpy().astype(np.int64)  # (N,)

        if self.flatten:
            self._x = self._x.reshape(len(self._x), -1)  # (N, C*H*W)

        if self.one_hot:
            self._y = self._one_hot_batch(self._y)  # (N, n_classes)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx):
        """
        Indexación O(1) — los datos ya están en RAM como NumPy arrays.
        Soporta int, slice, list y np.ndarray.
        """
        return self._x[idx], self._y[idx]

    def _one_hot_batch(self, y: np.ndarray) -> np.ndarray:
        oh = np.zeros((len(y), self.n_classes), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh
