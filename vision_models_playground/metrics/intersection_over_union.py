import torch


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
    x_min = torch.max(pred_box[..., 0], target_box[..., 0])
    y_min = torch.max(pred_box[..., 1], target_box[..., 1])
    x_max = torch.min(pred_box[..., 2], target_box[..., 2])
    y_max = torch.min(pred_box[..., 3], target_box[..., 3])

    intersection_area = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)
    pred_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    target_area = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])
    union_area = pred_area + target_area - intersection_area

    # Avoid division by zero
    zero_divisions = union_area == 0
    union_area[zero_divisions] = 1e-10
    intersection_area[zero_divisions] = 0

    return intersection_area / union_area


