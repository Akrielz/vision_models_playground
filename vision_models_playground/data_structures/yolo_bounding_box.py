from typing import Union

import torch

from vision_models_playground.datasets.yolo_pascal_voc_dataset import YoloPascalVocDataset
from vision_models_playground.metrics.intersection_over_union import compute_iou


class YoloBoundingBoxOperations:
    """
    This class is used to get the bounding boxes from the YOLO dataset.

    The standard tensor shape for the bounding boxes is (grid_size, grid_size, 5 * num_bounding_boxes + num_classes)
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

    def to_coords(self, bounding_boxes: torch.Tensor):
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
        self._check_input(bounding_boxes)

        x = self.get_attr(bounding_boxes, 'x')
        y = self.get_attr(bounding_boxes, 'y')
        w = self.get_attr(bounding_boxes, 'w')
        h = self.get_attr(bounding_boxes, 'h')

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
        pred_coords = self.to_coords(pred_bounding_boxes)
        target_coords = self.to_coords(target_bounding_boxes)

        return compute_iou(pred_coords, target_coords)


def main():
    voc_train = YoloPascalVocDataset(download=False)
    bbox = torch.stack([voc_train[0][1], voc_train[1][1], voc_train[2][1]])
    bbox_operations = YoloBoundingBoxOperations()
    to_coords = bbox_operations.compute_iou(bbox, bbox)
    print(to_coords.shape)


if __name__ == '__main__':
    main()