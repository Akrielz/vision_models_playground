from einops import rearrange
from matplotlib import pyplot as plt


def display_images_on_grid(images, n_rows=5, n_cols=5):
    """
    Display a grid of images.
    """
    plt.figure(figsize=(10, 10))

    channels = images.shape[1]

    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)

        image = rearrange(images[i], "c h w -> h w c")
        cmap = "gray" if channels == 1 else None

        plt.imshow(image, cmap=cmap)
        plt.axis('off')

    plt.show()


def display_image(image):
    """
    Display an image.
    """

    image = rearrange(image, "c h w -> h w c")
    channels = image.shape[0]
    cmap = "gray" if channels == 1 else None

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()