import torch
from einops import rearrange
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def to_autoencoder_dataset(dataset: torch.utils.data.Dataset):
    images = dataset.data

    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    return TensorDataset(images, images)


def get_mnist_dataset(root: str='./data'):
    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    return mnist_train, mnist_test


def get_cifar10_dataset(root: str='./data'):
    cifar10_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())

    return cifar10_train, cifar10_test


def get_tourism_dataset(root: str='./data', name: str='notredame'):
    photo_tour_train = datasets.PhotoTour(root=root, name=name, train=True, download=True)
    photo_tour_test = datasets.PhotoTour(root=root, name=name, train=False, download=True)

    return photo_tour_train, photo_tour_test


def get_n_images(dataset: torch.utils.data.Dataset, num_images: int):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    images = []
    for i, x in enumerate(train_loader):
        if i == num_images:
            break

        if isinstance(x, tuple):
            x = x[0]

        if len(x.shape) == 3:
            # Add batch_size dimension
            x = rearrange(x, "... -> 1 ...")

        images.append(x)

    images = torch.cat(images, dim=0)
    return images
