from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from vision_models_playground.datasets.yolo_pascal_voc_dataset import YoloPascalVocDataset
from vision_models_playground.datasets.yolo_pascal_voc_dataset_aug import YoloPascalVocDatasetAug


def to_autoencoder_dataset(dataset: torch.utils.data.Dataset):
    images = dataset.data

    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    return TensorDataset(images, images)


def get_mnist_dataset(
        root: str = './data',
        download: bool = False
):
    mnist_train = datasets.MNIST(root=root, train=True, download=download, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root=root, train=False, download=download, transform=transforms.ToTensor())

    return mnist_train, mnist_test


def get_cifar10_dataset(
        root: str = './data',
        download: bool = False
):
    cifar10_train = datasets.CIFAR10(root=root, train=True, download=download, transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root=root, train=False, download=download, transform=transforms.ToTensor())

    return cifar10_train, cifar10_test


def get_tourism_dataset(
        root: str = './data',
        name: str = 'notredame',
        download: bool = False
):
    photo_tour_train = datasets.PhotoTour(root=root, name=name, train=True, download=download)
    photo_tour_test = datasets.PhotoTour(root=root, name=name, train=False, download=download)

    return photo_tour_train, photo_tour_test


def get_voc_detection_dataset_raw(
        root: str = './data',
        year: str = '2012',
        download: bool = False
):
    voc_train = datasets.VOCDetection(root=root, year=year, image_set='train', download=download)
    voc_test = datasets.VOCDetection(root=root, year=year, image_set='val', download=download)

    return voc_train, voc_test


def get_voc_detection_dataset_yolo(
        root: str = './data',
        year: str = '2012',
        download: bool = False,
        num_bounding_boxes: int = 2,
        grid_size: int = 7
):
    kwargs = {
        'root': root,
        'year': year,
        'download': download,
        'num_bounding_boxes': num_bounding_boxes,
        'grid_size': grid_size
    }

    voc_train = YoloPascalVocDataset(phase='train', **kwargs)
    voc_test = YoloPascalVocDataset(phase='val', **kwargs)

    return voc_train, voc_test


def get_voc_detection_dataset_yolo_aug(
        root: str = './data',
        year: str = '2012',
        download: bool = False,
        num_bounding_boxes: int = 2,
        grid_size: int = 7,
        aug_transform: Optional[nn.Module] = None
):
    kwargs = {
        'root': root,
        'year': year,
        'download': download,
        'num_bounding_boxes': num_bounding_boxes,
        'aug_transform': aug_transform,
        'grid_size': grid_size
    }
    voc_train = YoloPascalVocDatasetAug(phase='train', **kwargs)
    voc_test = YoloPascalVocDatasetAug(phase='val', **kwargs)

    return voc_train, voc_test


def get_image_net_dataset(
        root: str = './data'
):
    image_net_train = datasets.ImageNet(root=root, split='train')
    image_net_test = datasets.ImageNet(root=root, split='val')

    return image_net_train, image_net_test


def get_n_images(dataset: torch.utils.data.Dataset, num_images: int):
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    images = []
    for i, x in enumerate(train_loader):
        if i == num_images:
            break

        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]

        if len(x.shape) == 3:
            # Add batch_size dimension
            x = rearrange(x, "... -> 1 ...")

        images.append(x)

    images = torch.cat(images, dim=0)
    return images
