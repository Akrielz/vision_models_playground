from typing import Tuple

import torch
from torchmetrics import Metric


def compute_intersection_and_union(
        pred_box: torch.Tensor,
        target_box: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the intersection and union areas between the predicted bounding boxes and the target bounding boxes.

    The boxes will have on the last dimension the coordinates (x_min, y_min, x_max, y_max).

    Arguments
    ---------
    pred_box: torch.Tensor
        The predicted bounding boxes with shape (batch_size, ..., 4)

    target_box: torch.Tensor
        The target bounding boxes with shape (batch_size, ..., 4)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The intersection and union areas with shape (batch_size, ...)
    """

    # Compute the intersection and union areas
    x_min = torch.max(pred_box[..., 0], target_box[..., 0])
    y_min = torch.max(pred_box[..., 1], target_box[..., 1])
    x_max = torch.min(pred_box[..., 2], target_box[..., 2])
    y_max = torch.min(pred_box[..., 3], target_box[..., 3])

    intersection_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)
    pred_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    target_area = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])
    union_area = pred_area + target_area - intersection_area

    return intersection_area, union_area


def compute_iou(
        pred_box: torch.Tensor,
        target_box: torch.Tensor
) -> torch.Tensor:
    """
    Get the Intersection over Union (IoU) between the predicted bounding boxes and the target bounding boxes.

    The boxes will have on the last dimension the coordinates (x_min, y_min, x_max, y_max).

    Arguments
    ---------
    pred_box: torch.Tensor
        The predicted bounding boxes with shape (batch_size, ..., 4)

    target_box: torch.Tensor
        The target bounding boxes with shape (batch_size, ..., 4)

    Returns
    -------
    torch.Tensor
        The IoU between the predicted bounding boxes and the target bounding boxes with shape (batch_size, ...)
    """

    # Compute the intersection and union areas
    intersection_area, union_area = compute_intersection_and_union(pred_box, target_box)

    # Avoid division by zero
    zero_divisions = union_area == 0
    union_area[zero_divisions] = 1e-10
    intersection_area[zero_divisions] = 0

    return intersection_area / union_area


class IntersectionOverUnion(Metric):
    """
    Intersection over Union (IoU) metric. It is computed as the ratio between the intersection and the union of the
    predicted bounding boxes and the target bounding boxes.
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("intersection", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred_box: torch.Tensor, target_box: torch.Tensor):
        """
        Update the metric state with the new predicted and target bounding boxes.

        Arguments
        ---------
        pred_box: torch.Tensor
            The predicted bounding boxes with shape (batch_size, ..., 4)

        target_box: torch.Tensor
            The target bounding boxes with shape (batch_size, ..., 4)
        """

        intersection_area, union_area = compute_intersection_and_union(pred_box, target_box)

        self.intersection += intersection_area.sum()
        self.union += union_area.sum()

    def compute(self):
        """
        Compute the IoU metric.

        Returns
        -------
        torch.Tensor
            The IoU metric with shape (1,)
        """

        # Avoid division by zero
        if self.union == 0:
            return torch.tensor(0)

        return self.intersection / self.union


def main():
    # Create the predicted and
    pred_box = torch.tensor([
        [0, 0, 10, 10],
        [0, 0, 10, 10],
    ], dtype=torch.float32)

    target_box = torch.tensor([
        [0, 0, 10, 10],
        [0, 0, 5, 5],
    ], dtype=torch.float32)

    # Apply the metric
    iou = IntersectionOverUnion()
    print(iou(pred_box, target_box))  # (0.25 + 1.0) / 2 = 0.625


if __name__ == "__main__":
    main()
