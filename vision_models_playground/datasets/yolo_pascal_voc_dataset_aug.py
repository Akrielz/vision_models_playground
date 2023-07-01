from typing import Tuple, Optional, Literal

from torch import nn
from torchvision.transforms import ToTensor

from vision_models_playground.datasets.yolo_pascal_voc_dataset import YoloPascalVocDataset
from vision_models_playground.transforms.auto import AutoTransformWithCoords
from vision_models_playground.transforms.compose import ComposeGeneral
from vision_models_playground.transforms.resize import ResizeWithCoords
from vision_models_playground.transforms.transform import WithCoords


class YoloPascalVocDatasetAug(YoloPascalVocDataset):
    """
    This is the augmented version of the YoloPascalVocDataset, which gives the same output as the original dataset, extended
    with the data augmented.

    Dataset for the Pascal VOC dataset, but with the necessary transformations to be used with YOLOv1 according to the paper
    https://arxiv.org/pdf/1506.02640.pdf

    The images are resized to image_size, and they are divided in a grid of grid_size x grid_size cells.
    Each cell has num_bounding_boxes bounding boxes, and each bounding box has 5 values:
    [x, y, w, h, confidence], where x and y are the relative coordinates of the obj compared with the cell,
    w and h are the relative width and height of the obj, and confidence is the confidence that the obj is in the cell.

    The labels are encoded as follows:
    [grid_size, grid_size, num_bounding_boxes, 5 + num_classes]

    For example for grid_size=7, bounding boxes = 2 and classes = 20, we would have the shape:
    [7, 7, 2*5 + 20] = [7, 7, 30]
    Where we have the first two bounding boxes, and then the classes on the last dimension.
    [(x, y, w, h, confidence), (x, y, w, h, confidence), class_1, class_2, ..., class_20]
    """

    def __init__(
            self,
            root: str = './data',
            phase: Literal['train', 'val'] = 'train',
            year: str = '2012',
            image_size: Tuple[int, int] = (448, 448),
            num_bounding_boxes: int = 2,
            grid_size: int = 7,
            download: bool = False,
            aug_transform: Optional[nn.Module] = None,
            aug_data_multiplier: int = 1
    ):
        super().__init__(
            root=root,
            phase=phase,
            year=year,
            image_size=image_size,
            num_bounding_boxes=num_bounding_boxes,
            grid_size=grid_size,
            download=download,
        )

        # Default init for aug transform
        if aug_transform is None:
            aug_transform = AutoTransformWithCoords(size=image_size)

        aug_transform = ComposeGeneral([
            WithCoords(ToTensor()),
            aug_transform,
            ResizeWithCoords(image_size)
        ])

        # Save the original transform and the augmentation transform
        self.normal_transform = self.transform
        self.augment_transform = aug_transform

        # Compute the total length of the dataset
        self.normal_len = super().__len__()
        self.aug_len = self.normal_len * aug_data_multiplier
        self.total_len = self.normal_len + self.aug_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        # Prepare the correct info
        transform = self.normal_transform if idx < self.normal_len else self.augment_transform
        raw_idx = idx % self.normal_len

        # Compute the labels
        image, target = self.raw_dataset[raw_idx]
        image, target = self._apply_transform(image, target, transform)
        labels = self._compute_labels(target)

        return image, labels
