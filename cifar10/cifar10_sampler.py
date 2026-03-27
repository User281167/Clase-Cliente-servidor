import tempfile

import numpy as np
import torchvision
import torchvision.transforms as transforms


class Cifar10DataSampler:
    """
    Dataset sampler para CIFAR-10.

    Cargar los datos de manera eficiente utilizando torchvision.
    Woker carga todo el dataset en memoria antes de empezar a entrenar.

    Servidor envia epoch, rank, batch_size y num_workers al cliente.
    El cliente utiliza estos valores para configurar el dataloader.
    Pytorch se encarga de hacer shuffle y evitar que se solapen los batches.
    """

    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        grayscale: bool = True,
        flatten: bool = True,
        one_hot: bool = True,
        n_classes: int = 10,
    ):
        """
        Inicializa el dataset sampler para CIFAR-10.

        Args:
            root (str | None): Directorio donde se guardarán los datos. Si es None, se usa un directorio temporal.
            train (bool): Si es True, carga el conjunto de entrenamiento; si es False, carga el conjunto de prueba.
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

        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retorna el item en el índice o lista `idx` del dataset.
        Convierte los datos a numpy antes de retornarlos.
        """
        # convertir numpy scalars a int
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)

        if isinstance(idx, int):
            x, y = self.dataset[idx]
            x_np = np.array(x, dtype=np.float32)

            if self.flatten:
                x_np = x_np.flatten()
            if self.one_hot:
                y = self.one_hot_encode(y)

            return x_np, y
        elif (
            isinstance(idx, slice)
            or isinstance(idx, list)
            or isinstance(idx, np.ndarray)
        ):
            xs, ys = [], []

            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self.dataset)))
            else:
                indices = idx

            for i in indices:
                x, y = self.dataset[i]
                x_np = np.array(x, dtype=np.float32)

                if self.flatten:
                    x_np = x_np.flatten()
                if self.one_hot:
                    y = self.one_hot_encode(y)

                xs.append(x_np)
                ys.append(y)
            return np.stack(xs), np.stack(ys)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def one_hot_encode(self, y: int | np.ndarray) -> np.ndarray:
        if isinstance(y, int):
            vec = np.zeros(self.n_classes, dtype=np.float32)
            vec[y] = 1.0
            return vec
        else:
            oh = np.zeros((len(y), self.n_classes), dtype=np.float32)
            oh[np.arange(len(y)), y] = 1.0
            return oh
