from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def get_mnist_dataloader(
        root: str = "data/mnist",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        train: bool = True,
        test: bool = False,
        download: bool = True
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    train_dataloader = None
    test_dataloader = None
    mean = (0.5, )
    std = (0.5, )

    print("getting MNIST dataloader...")
    if train:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.MNIST(
            root = root,
            train = True,
            download = download,
            transform = train_transform
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

    if test:
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_dataset = datasets.MNIST(
            root = root,
            train = False,
            download = download,
            transform = test_transform
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

    print("MNIST dataloader done...\n")
    return (train_dataloader, test_dataloader)


def get_cifar10_dataloader(
        root: str = "data/cifar10",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        train: bool = True,
        test: bool = False,
        download: bool = True
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    train_dataloader = None
    test_dataloader = None
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    print("getting CIFAR10 dataloader...")
    if train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.CIFAR10(
            root = root,
            train = True,
            download = download,
            transform = train_transform
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

    if test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_dataset = datasets.CIFAR10(
            root = root,
            train = False,
            download = download,
            transform = test_transform
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

    print("CIFAR10 dataloader done...\n")
    return (train_dataloader, test_dataloader)
