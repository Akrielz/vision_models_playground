from matplotlib import pyplot as plt
from torch import nn

from torchvision.transforms import RandomHorizontalFlip, RandomPerspective, RandomResizedCrop, RandomErasing, RandomAffine, \
    RandomRotation, ColorJitter

from vision_models_playground.utility.datasets import get_mnist_dataset, get_n_images


class Augmeneter(nn.Module):
    def __init__(
            self,
            image_size: int = 32,
            background_color: float = 1.0,
            rotation_angles: int = 30
    ):
        super().__init__()

        self.net = nn.Sequential(
            RandomHorizontalFlip(),
            RandomPerspective(fill=background_color),
            RandomResizedCrop(size=image_size, scale=(0.90, 1.00)),
            RandomErasing(value=background_color, scale=(0.1, 0.15)),
            RandomAffine(degrees=rotation_angles, fill=background_color),
            RandomRotation(degrees=rotation_angles, fill=background_color),
            ColorJitter(brightness=(0.8, 1.0)),
            ColorJitter(contrast=(0.8, 1.0)),
            ColorJitter(saturation=(0.5, 1.0)),
        )

    def forward(self, x):
        return self.net(x)


def main():
    augmenter = Augmeneter(image_size=28, background_color=0.0, rotation_angles=15)
    mnist_dataset_train, _ = get_mnist_dataset()
    images = get_n_images(mnist_dataset_train, num_images=25)
    images = augmenter(images)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].cpu().data.numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()