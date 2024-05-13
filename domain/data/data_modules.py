import os

import lightning as L
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from domain.data.transforms import SimpleFreqSpace, SimpleComplex2Vec


seed = torch.Generator().manual_seed(42)


class BaseDataModule(L.LightningDataModule):
    def __init__(self, domain: str, batch_size: int):
        super().__init__()
        self.domain = domain
        self.batch_size = batch_size

        if self.domain == 'freq':
            self.domain_transform = transforms.Compose(
                [SimpleFreqSpace(), SimpleComplex2Vec()]
            )
        else:
            self.domain_transform = torch.nn.Identity()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates Dataloader for training phase.

        Returns:
            Dataloader for training phase.
        """
        return torch.utils.data.DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Creates Dataloader for validation phase.

        Returns:
            Dataloader for validation phase.
        """
        return torch.utils.data.DataLoader(self.val_set, self.batch_size)


class ImageNetDataModule(BaseDataModule):
    def __init__(self, data_dir: str, input_domain: str, batch_size: int = 32) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.input_domain = input_domain
        self.batch_size = batch_size

    def setup(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'val')

        self.train_set = ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    self.domain_transfrom,
                ]
            ),
        )

        self.val_set = ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                    self.domain_transfrom,
                ]
            ),
        )


class MNISTDataModule(BaseDataModule):
    def __init__(self, domain: str, batch_size: int = 32) -> None:
        super().__init__(domain=domain, batch_size=batch_size)
        self.image_shape = (28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.MNIST(root='MNIST', download=True, train=True)
        datasets.MNIST(root='MNIST', download=True, train=False)

    def setup(self, stage: str):
        tensor_transform = transforms.ToTensor()

        self.test_set = datasets.MNIST(
            root='MNIST',
            download=True,
            train=False,
            transform=transforms.Compose([tensor_transform, self.domain_transform]),
        )

        data_set = datasets.MNIST(
            root='MNIST',
            download=True,
            train=True,
            transform=transforms.Compose([tensor_transform, self.domain_transform]),
        )

        # use 20% of training data for validation
        train_set_size = int(len(data_set) * 0.8)
        valid_set_size = len(data_set) - train_set_size

        self.train_set, self.val_set = torch.utils.data.random_split(
            data_set, [train_set_size, valid_set_size], generator=seed
        )


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, domain: str, batch_size: int = 4) -> None:
        super().__init__(domain=domain, batch_size=batch_size)
        self.image_shape = (32, 32, 3)
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.CIFAR10(root='CIFAR10', download=True, train=True)
        datasets.CIFAR10(root='CIFAR10', download=True, train=False)

    def setup(self, stage: str):
        cifar10_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.test_set = datasets.CIFAR10(
            root='CIFAR10',
            download=True,
            train=False,
            transform=transforms.Compose([cifar10_transform, self.domain_transform]),
        )

        data_set = datasets.CIFAR10(
            root='CIFAR10',
            download=True,
            train=True,
            transform=transforms.Compose([cifar10_transform, self.domain_transform]),
        )

        # use 20% of training data for validation
        train_set_size = int(len(data_set) * 0.8)
        valid_set_size = len(data_set) - train_set_size

        self.train_set, self.val_set = torch.utils.data.random_split(
            data_set, [train_set_size, valid_set_size], generator=seed
        )
