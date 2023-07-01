import os
import pickle
from typing import Literal, Tuple, Dict, Any, List

import torch
from PIL.Image import Image
from einops import repeat
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCDetection

from vision_models_playground.transforms.compose import ComposeGeneral
from vision_models_playground.transforms.resize import ResizeWithCoords
from vision_models_playground.transforms.transform import WithCoords


class YoloPascalVocDataset(Dataset):
    """
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
            download: bool = False
    ):
        """
        Arguments
        ---------

        root: str = './data'
            Root directory where the dataset is stored

        phase: Literal['train', 'val'] = 'train'
            Phase of the dataset to use

        year: str = '2012'
            Year of the dataset to use

        image_size: Tuple[int, int] = (448, 448)
            Size of the images to use. This is [height, width]

        num_bounding_boxes: int = 2
            Number of bounding boxes per cell

        grid_size: int = 7
            Size of the grid to use

        download: bool = False
            Whether to download the dataset if it is not found
        """

        super().__init__()

        # Save the data
        self.root = root
        self.phase = phase
        self.year = year
        self.image_size = image_size
        self.num_bounding_boxes = num_bounding_boxes
        self.grid_size = grid_size
        self.download = download

        # Save the paths
        self.class_map_file = f'{root}/voc_classes/{phase}_{year}.pkl'

        # Create the transform, but don't apply it yet because the VocDetection dataset does not
        # work with transforms, so we will apply it manually
        self.transform = ComposeGeneral([
            WithCoords(ToTensor()),
            ResizeWithCoords(image_size, antialias=True)
        ])
        self.raw_dataset = VOCDetection(root=root, year=year, image_set=phase, download=download)

        # Get the class map
        self.class_map = self._get_class_map()

        # Prepare the shapes
        self.class_shape = (self.grid_size, self.grid_size, self.num_classes)
        self.bounding_box_shape = (self.grid_size, self.grid_size, 5 * self.num_bounding_boxes)
        self.target_shape = (self.grid_size, self.grid_size, 5 * self.num_bounding_boxes + self.num_classes)

    @property
    def num_classes(self):
        return len(self.class_map)

    @property
    def classes(self):
        return list(self.class_map.keys())

    def _get_class_map(self):
        # Check if the class_file exists
        if os.path.exists(self.class_map_file):
            # Read it with yaml
            with open(self.class_map_file, 'rb') as f:
                class_map = pickle.load(f)
            return class_map

        # If it doesn't exist, create it
        class_set = set()
        for image, target in self.raw_dataset:
            for obj in target['annotation']['object']:
                class_set.add(obj['name'])

        # Sort the classes so that the class_map is always the same
        class_list = sorted(list(class_set))
        class_map = {class_list[i]: i for i in range(len(class_list))}

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.class_map_file), exist_ok=True)

        # Save it with pickle
        with open(self.class_map_file, 'wb') as f:
            pickle.dump(class_map, f)

        return class_map

    def __len__(self):
        return len(self.raw_dataset)

    @staticmethod
    def _extract_objects_from_target(target: Dict) -> Tuple[List[Tuple], List[str]]:
        """
        Extracts the objects from the target dictionary
        """

        bboxes = []
        classes = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            x_min = int(bbox['xmin'])
            y_min = int(bbox['ymin'])
            x_max = int(bbox['xmax'])
            y_max = int(bbox['ymax'])
            class_name = obj['name']

            bboxes.append((x_min, y_min, x_max, y_max))
            classes.append(class_name)

        return bboxes, classes

    @staticmethod
    def _bboxes_to_coords(bboxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """
        Converts the objects to coordinates
        """
        coords = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            coords.append([x_min, y_min])
            coords.append([x_min, y_max])
            coords.append([x_max, y_min])
            coords.append([x_max, y_max])

        return torch.tensor(coords)

    @staticmethod
    def _coords_to_bboxes(coords: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """
        Converts the coordinates to objects
        """
        bboxes = []
        for i in range(0, len(coords), 4):
            x_min = coords[i:i+4, 0].min()
            y_min = coords[i:i+4, 1].min()
            x_max = coords[i:i+4, 0].max()
            y_max = coords[i:i+4, 1].max()
            bboxes.append((x_min, y_min, x_max, y_max))

        # Cast all the bboxes to int
        bboxes = [(int(x_min), int(y_min), int(x_max), int(y_max)) for x_min, y_min, x_max, y_max in bboxes]

        return bboxes

    def _bboxes_and_class_to_objects(self, bboxes: List[Tuple], class_names: List[str]) -> List[Dict[str, int]]:
        objects = []
        for bbox, class_name in zip(bboxes, class_names):
            class_id = self.class_map[class_name]
            obj = {
                'x_min': bbox[0],
                'y_min': bbox[1],
                'x_max': bbox[2],
                'y_max': bbox[3],
                'class_id': class_id,
            }
            objects.append(obj)

        return objects

    def _apply_transform(
            self,
            image: Image,
            target: Dict[str, Any],
            transform: nn.Module
    ) -> Tuple[torch.Tensor, List[Dict[str, int]]]:
        """
        Applies the transform to the image and target
        """

        # Extract the objects from the target
        bboxes, classes = self._extract_objects_from_target(target)

        # Convert the objects to coordinates
        coords = self._bboxes_to_coords(bboxes)

        # Apply the transform to the image and coordinates
        image, coords = transform(image, coords)

        # Convert the coordinates back to objects
        bboxes = self._coords_to_bboxes(coords)

        # Convert the objects to the target format
        objects = self._bboxes_and_class_to_objects(bboxes, classes)

        return image, objects

    @property
    def cell_size(self):
        """
        Returns the size of each cell in the grid as a tuple (grid_size_y, grid_size_x)
        """
        return self.image_size[0] // self.grid_size, self.image_size[1] // self.grid_size

    def __getitem__(self, idx: int):
        image, target = self.raw_dataset[idx]
        image, target = self._apply_transform(image, target, self.transform)
        labels = self._compute_labels(target)

        return image, labels

    def _compute_labels(self, target: List[Dict[str, int]]):
        # Get the grid size
        grid_size_y, grid_size_x = self.cell_size

        # Get the boxes and labels
        class_labels = torch.zeros(self.class_shape)
        bounding_box_labels = torch.zeros(self.bounding_box_shape)

        bbox_grid_index = {}  # (row, col) -> (bbox_index)
        class_grid_ids = {}  # (row, col) -> (class_id)

        for i, box in enumerate(target):
            # Compute the area of the box
            area = (box['x_max'] - box['x_min']) * (box['y_max'] - box['y_min'])

            if area < 1:
                continue

            # Get the class id
            class_id = box['class_id']

            # Add the label to the list
            mid_x = (box['x_min'] + box['x_max']) // 2
            mid_y = (box['y_min'] + box['y_max']) // 2

            # Compute the grid cell coordinates
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            # Check if the grid cell is already occupied, and if it is make sure that the class is the same
            if (row, col) in class_grid_ids and class_grid_ids[(row, col)] != class_id:
                # If there is already a bounding box in this grid cell, but it is a different class,
                # then skip this bounding box
                continue

            # Compute one hot encoding for the class
            class_one_hot = torch.zeros(self.num_classes)
            class_one_hot[class_id] = 1.0
            class_labels[row, col] = class_one_hot
            class_grid_ids[(row, col)] = class_id

            # Compute the bounding box index
            bbox_index = bbox_grid_index.get((row, col), 0)

            if bbox_index >= self.num_bounding_boxes:
                # If there are already num_bounding_boxes bounding boxes in this grid cell, then skip this bounding box
                continue

            start_idx = 5 * bbox_index

            # Compute the bounding box
            bbox_info = torch.tensor([
                (mid_x - col * grid_size_x) / self.image_size[0],  # x coord relative to grid cell relative to image
                (mid_y - row * grid_size_y) / self.image_size[1],  # y coord relative to grid cell relative to image
                (box['x_max'] - box['x_min']) / self.image_size[0],  # Width relative to image
                (box['y_max'] - box['y_min']) / self.image_size[1],  # Height relative to image
                1.0  # Confidence
            ])

            # Repeat the bounding box such that it avoids overwriting the previous one intel, but also
            # doesn't let zombie bounding boxes linger around
            bounding_box_labels[row, col, start_idx:] = repeat(bbox_info, 'i -> (j i)', j=self.num_bounding_boxes - bbox_index)
            bbox_grid_index[(row, col)] = bbox_index + 1

        labels = torch.cat([bounding_box_labels, class_labels], dim=-1)
        return labels


def main():
    dataset = YoloPascalVocDataset()
    print(dataset[0])


if __name__ == '__main__':
    main()