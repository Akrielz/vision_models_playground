from typing import Union

import torch

from vision_models_playground.metrics.intersection_over_union import compute_iou


class YoloBoundingBoxOperations:
    """
    This class is used to get the bounding boxes from the YOLO dataset.

    The standard tensor shape for the bounding boxes is (grid_size, grid_size, 5 * num_bounding_boxes + num_classes)
    For example for grid_size=7, bounding boxes = 2 and classes = 20, we would have the shape:
    [7, 7, 2*5 + 20] = [7, 7, 30]
    Where we have the first two bounding boxes, and then the classes on the last dimension.
    [(x, y, w, h, confidence), (x, y, w, h, confidence), class_1, class_2, ..., class_20]
    """
    def __init__(
            self,
            num_bounding_boxes: int = 2,
            num_classes: int = 20
    ):
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes

        self.dim = 5 * num_bounding_boxes + num_classes
        self.start_index_classes = 5 * num_bounding_boxes

        self.attribute_map = {
            'x': 0,
            'y': 1,
            'w': 2,
            'h': 3,
            'confidence': 4,
            'c': 4
        }

    def _check_input(
            self,
            bounding_boxes: torch.Tensor
    ):
        assert bounding_boxes.shape[-1] == self.dim, f"Expected last dimension to be {self.dim}, but got {bounding_boxes.shape[-1]}"

    def get_classes(
            self,
            bounding_boxes: torch.Tensor
    ) -> torch.Tensor:
        self._check_input(bounding_boxes)
        return bounding_boxes[..., self.start_index_classes:]

    def get_bounding_boxes(
            self,
            bounding_boxes: torch.Tensor
    ) -> torch.Tensor:
        self._check_input(bounding_boxes)
        return bounding_boxes[..., :self.start_index_classes]

    def get_attr(
            self,
            bounding_boxes: torch.Tensor,
            attr: Union[str, int]
    ) -> torch.Tensor:
        """
        Get the bounding boxes attributes for all the bounding boxes in the tensor.

        Arguments
        ---------
        bounding_boxes: torch.Tensor
            The bounding boxes tensor with shape (grid_size, grid_size, 5 * num_bounding_boxes + num_classes)

        attr: Union[str, int]
            The attribute to get. It can be either a string or an integer.

            If attr = 0 or attr = 'x', then it will return the x coordinate of the bounding boxes.
            If attr = 1 or attr = 'y', then it will return the y coordinate of the bounding boxes.
            If attr = 2 or attr = 'w', then it will return the width of the bounding boxes.
            If attr = 3 or attr = 'h', then it will return the height of the bounding boxes.
            If attr = 4 or attr = 'confidence' or attr = 'c', then it will return the confidence of the bounding boxes.

        Returns
        -------
        torch.Tensor
            The bounding boxes attributes with shape (grid_size, grid_size, num_bounding_boxes)
        """

        if isinstance(attr, str):
            attr = self.attribute_map[attr]

        if attr < 0 or attr >= 5:
            raise ValueError(f"Expected attr to be between 0 and 4, but got {attr}")

        self._check_input(bounding_boxes)

        bounding_boxes = self.get_bounding_boxes(bounding_boxes)
        return bounding_boxes[..., attr::5]

    def compute_confidence_mask(
            self,
            bounding_boxes: torch.Tensor,
    ):
        confidence = self.get_attr(bounding_boxes, 'confidence')
        return confidence >= 0.5

    def to_corners(self, bounding_boxes: torch.Tensor):
        """
        Convert the bounding boxes to coordinates.

        Arguments
        ---------
        bounding_boxes: torch.Tensor
            The bounding boxes tensor with shape (grid_size, grid_size, 5 * num_bounding_boxes + num_classes)

        Returns
        -------
        torch.Tensor
            The bounding boxes coordinates with shape (grid_size, grid_size, num_bounding_boxes, 4)
        """
        x, y, w, h = torch.unbind(self.get_window(bounding_boxes), dim=-1)

        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2

        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def get_window(self, bounding_boxes: torch.Tensor):
        self._check_input(bounding_boxes)
        x = self.get_attr(bounding_boxes, 'x')
        y = self.get_attr(bounding_boxes, 'y')
        w = self.get_attr(bounding_boxes, 'w')
        h = self.get_attr(bounding_boxes, 'h')

        return torch.stack([x, y, w, h], dim=-1)

    def get_window_for_yolo_loss(self, bounding_boxes: torch.Tensor):
        x = self.get_attr(bounding_boxes, 'x')
        y = self.get_attr(bounding_boxes, 'y')
        w = self.get_attr(bounding_boxes, 'w')
        h = self.get_attr(bounding_boxes, 'h')

        w = torch.sign(w) * torch.sqrt(torch.abs(w)) + 1e-6
        h = torch.sign(h) * torch.sqrt(torch.abs(h)) + 1e-6

        return torch.stack([x, y, w, h], dim=-1)

    def get_window_absolute(self, bounding_boxes: torch.Tensor):
        self._check_input(bounding_boxes)
        x = self.get_attr(bounding_boxes, 'x')
        y = self.get_attr(bounding_boxes, 'y')
        w = self.get_attr(bounding_boxes, 'w')
        h = self.get_attr(bounding_boxes, 'h')

        # the above values are relative to the grid cell
        # we need to convert them to absolute values
        grid_shape = bounding_boxes.shape[-3:-1]
        grid_height = grid_shape[0]
        grid_width = grid_shape[1]

        cell_height = 1 / grid_shape[0]
        cell_width = 1 / grid_shape[1]

        cell_indexes = torch.stack(torch.meshgrid(torch.arange(grid_height), torch.arange(grid_width), indexing='ij'), dim=-1)
        cell_indexes = cell_indexes.to(bounding_boxes.device)

        offset_x = cell_indexes[..., 1] * cell_width
        offset_y = cell_indexes[..., 0] * cell_height

        offset_x = offset_x.unsqueeze(-1)
        offset_y = offset_y.unsqueeze(-1)

        x = x + offset_x
        y = y + offset_y

        return torch.stack([x, y, w, h], dim=-1)

    def to_corners_absolute(self, bounding_boxes: torch.Tensor):
        x, y, w, h = torch.unbind(self.get_window_absolute(bounding_boxes), dim=-1)

        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2

        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def compute_iou(
            self,
            pred_bounding_boxes: torch.Tensor,
            target_bounding_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the Intersection over Union (IoU) between the predicted bounding boxes and the target bounding boxes.

        Arguments
        ---------
        pred_bounding_boxes: torch.Tensor
            The predicted bounding boxes with shape (batch_size, grid_size, grid_size, 5 * num_bounding_boxes + num_classes)

        target_bounding_boxes: torch.Tensor
            The target bounding boxes with shape (batch_size, grid_size, grid_size, 5 * num_bounding_boxes + num_classes)

        Returns
        -------
        torch.Tensor
            The IoU between the predicted bounding boxes and the target bounding boxes with shape (grid_size, grid_size, num_bounding_boxes)
        """
        self._check_input(pred_bounding_boxes)
        self._check_input(target_bounding_boxes)

        # Convert the bounding boxes to coordinates per box [batch_size, grid_size, grid_size, num_bounding_boxes, 4]
        pred_coords = self.to_corners(pred_bounding_boxes)
        target_coords = self.to_corners(target_bounding_boxes)

        return compute_iou(pred_coords, target_coords)