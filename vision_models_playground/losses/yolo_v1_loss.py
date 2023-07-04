from typing import Literal

import torch
from torch import nn

from vision_models_playground.data_structures.yolo_bounding_box import YoloBoundingBoxOperations
from vision_models_playground.datasets.yolo_pascal_voc_dataset import YoloPascalVocDataset


class YoloV1Loss(nn.Module):
    """
    The Sum Squared Error used in the YOLO v1 paper:
    https://arxiv.org/pdf/1506.02640.pdf

    The loss is calculated as follows:

    - For each cell, For each bounding box if the cell has an object:
    obj_loss = weight_coords * (
        [x_i - x_hat_i]^2 + [y_i - y_hat_i]^2 +
        [sqrt(w_i) - sqrt(w_hat_i)]^2 + [sqrt(h_i) - sqrt(h_hat_i)]^2
    ) + [C_i - C_hat_i]^2

    - For each cell, For each bounding box if the cell does not have an object:
    no_obj_loss = weight_noobj * [C_i - C_hat_i]^2

    - For each cell if the cell has an object:
    class_loss = [p_i - p_hat_i]^2

    And the total loss is the sum of all the scores.
    total_loss = obj_loss + no_obj_loss + class_loss

    Where:
    - x_i, y_i, w_i, h_i are the ground truth values for the bounding box
    - x_hat_i, y_hat_i, w_hat_i, h_hat_i are the predicted values for the bounding box
    - C_i is the ground truth confidence
    - C_hat_i is the predicted confidence
    - p_i is the ground truth class
    - p_hat_i is the predicted class
    - weight_coords is the weight for the loss of the cells that have an object applied only for coords
    - weight_no_obj is the weight for the loss of the cells that do not have an object
    """

    def __init__(
            self,
            weight_coord: float = 5.0,
            weight_obj: float = 1.0,
            weight_no_obj: float = 0.5,
            num_bounding_boxes: int = 2,
            num_classes: int = 20,
            reduction: Literal['mean', 'sum'] = 'mean',
    ):
        super().__init__()

        self.weight_coord = weight_coord
        self.weight_no_obj = weight_no_obj
        self.weight_obj = weight_obj
        self.bb_ops = YoloBoundingBoxOperations(
            num_bounding_boxes=num_bounding_boxes,
            num_classes=num_classes
        )

        self.mse_sum = nn.MSELoss(reduction='sum')

        # I know the paper proposes mse for classes too, but I think this is better
        self.ce_sum = nn.CrossEntropyLoss(reduction='sum')

        self.reduction = reduction

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        """
        Calculate the loss between the predicted and target tensors.

        Arguments
        ---------
        predicted : torch.Tensor
            The predicted tensor with shape [batch_size, grid_size, grid_size, 5 * num_bounding_boxes + num_classes]

        target : torch.Tensor
            The target tensor with shape [batch_size, grid_size, grid_size, 5 * num_bounding_boxes + num_classes]

        Returns
        -------
        torch.Tensor
            The loss between the predicted and target tensors.
        """

        assert predicted.shape == target.shape, \
            f'Predicted and target tensors must have the same shape. Got {predicted.shape} and {target.shape}'

        # Save batch_size info
        batch_size = predicted.shape[0]

        # Compute a mask for the cells that have an object
        mask_obj_per_box = self.bb_ops.compute_confidence_mask(target)  # Shape: [batch_size, grid_size, grid_size, num_bounding_boxes]
        mask_no_obj_per_box = ~mask_obj_per_box  # Shape: [batch_size, grid_size, grid_size, num_bounding_boxes]

        # If we have at least one bounding box with an object in a cell, then the cell has an object
        mask_obj = mask_obj_per_box.any(dim=-1)  # Shape: [batch_size, grid_size, grid_size]

        # Get the coords of the bounding boxes
        mask_box = mask_obj_per_box[mask_obj]
        x_pred, y_pred, w_pred, h_pred = torch.unbind(
            self.bb_ops.get_window_for_yolo_loss(predicted[mask_obj])[mask_box],
            dim=-1
        )  # Shape: [mask.sum()]
        x_target, y_target, w_target, h_target = torch.unbind(
            self.bb_ops.get_window_for_yolo_loss(target[mask_obj])[mask_box],
            dim=-1
        )  # Shape: [mask.sum()]

        # Get the confidence of the bounding boxes for pred
        confidence_pred = self.bb_ops.get_attr(predicted, 'confidence')  # Shape: [batch_size, grid_size, grid_size, num_bounding_boxes]
        with torch.no_grad():
            confidence_target = self.bb_ops.compute_iou(predicted, target)  # Shape: [batch_size, grid_size, grid_size, num_bounding_boxes]

        # Get the class of the bounding boxes
        class_pred = self.bb_ops.get_classes(predicted)  # Shape: [batch_size, grid_size, grid_size, num_classes]
        class_target = self.bb_ops.get_classes(target)  # Shape: [batch_size, grid_size, grid_size, num_classes]

        # Compute the loss for the cells that have an object
        bbox_loss = self.weight_coord * (
            self.mse_sum(x_pred, x_target) +
            self.mse_sum(y_pred, y_target) +
            self.mse_sum(w_pred, w_target) +
            self.mse_sum(h_pred, h_target)
        )

        # Compute the confidence loss
        # Here I've added weight_obj which by default is 1.0, therefore it doesn't change anything
        confidence_loss = (
            self.weight_obj * self.mse_sum(confidence_pred[mask_obj_per_box], confidence_target[mask_obj_per_box]) +
            self.weight_no_obj * self.mse_sum(confidence_pred[mask_no_obj_per_box], confidence_target[mask_no_obj_per_box])
        )

        # Compute the class loss
        class_loss = self.ce_sum(class_pred[mask_obj], class_target[mask_obj])

        # Compute the total loss
        total_loss = bbox_loss + confidence_loss + class_loss

        if self.reduction == 'mean':
            total_loss = total_loss / batch_size

        return total_loss


def main():
    voc_train = YoloPascalVocDataset(download=False)
    bbox_pred = torch.stack([voc_train[0][1], voc_train[1][1]])
    bbox_target = torch.stack([voc_train[2][1], voc_train[3][1]])

    loss = YoloV1Loss()
    loss(bbox_pred, bbox_target)


if __name__ == '__main__':
    main()
