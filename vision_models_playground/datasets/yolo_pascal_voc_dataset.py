import os
import pickle
from typing import Literal, Tuple, Dict, Any, List

import torch
from einops import repeat
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import VOCDetection


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
        self.transform = Compose([ToTensor(), Resize(image_size)])
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

    def _get_bounding_boxes(
            self,
            target: Dict[str, Any]
    ) -> List[Tuple[Dict[str, int], int]]:
        original_width = int(target['annotation']['size']['width'])
        original_height = int(target['annotation']['size']['height'])
        x_scale = self.image_size[0] / original_width
        y_scale = self.image_size[1] / original_height

        boxes = []
        for obj in target['annotation']['object']:
            box = obj['bndbox']
            coords = {
                'x_min': int(float(box['xmin']) * x_scale),
                'y_min': int(float(box['ymin']) * y_scale),
                'x_max': int(float(box['xmax']) * x_scale),
                'y_max': int(float(box['ymax']) * y_scale),
            }
            class_name = obj['name']
            class_id = self.class_map[class_name]
            boxes.append((coords, class_id))

        return boxes

    @property
    def cell_size(self):
        return self.image_size[0] // self.grid_size, self.image_size[1] // self.grid_size

    def __getitem__(self, idx: int):
        image, target = self.raw_dataset[idx]
        image = self.transform(image)

        grid_size_x, grid_size_y = self.cell_size

        # Get the boxes and labels
        class_labels = torch.zeros(self.class_shape)
        bounding_box_labels = torch.zeros(self.bounding_box_shape)

        bbox_grid_index = {}  # (row, col) -> (bbox_index)
        class_grid_ids = {}  # (row, col) -> (class_id)
        for i, (box, class_id) in enumerate(self._get_bounding_boxes(target)):
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
            start_idx = 5 * bbox_index

            # Compute the bounding box
            bbox_info = torch.tensor([
                (mid_x - col * grid_size_x) / self.image_size[0],     # x coord relative to grid cell relative to image
                (mid_y - row * grid_size_y) / self.image_size[1],     # y coord relative to grid cell relative to image
                (box['x_max'] - box['x_min']) / self.image_size[0],   # Width relative to image
                (box['y_max'] - box['y_min']) / self.image_size[1],   # Height relative to image
                1.0                                                   # Confidence
            ])

            # Repeat the bounding box such that it avoids overwriting the previous one intel, but also
            # doesn't let zombie bounding boxes linger around
            bounding_box_labels[row, col, start_idx:] = repeat(bbox_info, 'i -> (j i)', j=self.num_bounding_boxes - bbox_index)
            bbox_grid_index[(row, col)] = bbox_index + 1

        labels = torch.cat([bounding_box_labels, class_labels], dim=-1)

        return image, labels


def main():
    dataset = YoloPascalVocDataset()
    print(dataset[0])


if __name__ == '__main__':
    main()