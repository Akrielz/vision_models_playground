import torch
import torchmetrics
from einops import repeat
from torch import argmax
from torchmetrics.detection import MeanAveragePrecision

import vision_models_playground.data_structures.yolo_bounding_box as yolo_bounding_box


class YoloV1ClassMetricWrapper(torchmetrics.Metric):
    def __init__(
            self,
            metric: torchmetrics.Metric,
            num_bounding_boxes: int = 2,
            num_classes: int = 20
    ):
        super().__init__()
        self.metric = metric
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes
        self.bb_ops = yolo_bounding_box.YoloBoundingBoxOperations(
            num_bounding_boxes=num_bounding_boxes,
            num_classes=num_classes
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        class_pred = self.bb_ops.get_classes(preds)
        class_target = self.bb_ops.get_classes(target)

        mask_obj_per_box = self.bb_ops.compute_confidence_mask(target)
        mask_obj = mask_obj_per_box.any(dim=-1)

        class_pred = class_pred[mask_obj]
        class_target = class_target[mask_obj]

        class_target = argmax(class_target, dim=-1)

        self.metric.update(class_pred, class_target)

    def compute(self):
        return self.metric.compute()

    def __repr__(self):
        return self.metric.__repr__()


class YoloV1MeanAveragePrecision(torchmetrics.Metric):
    def __init__(
            self,
            num_bounding_boxes: int = 2,
            num_classes: int = 20
    ):
        super().__init__()
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes
        self.bb_ops = yolo_bounding_box.YoloBoundingBoxOperations(
            num_bounding_boxes=num_bounding_boxes,
            num_classes=num_classes
        )

        self.mAP = MeanAveragePrecision(num_classes=num_classes)

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        predicted_output, target_output = self.convert_to_expected_format(predicted, target)
        self.mAP.update(predicted_output, target_output)

    def convert_to_expected_format(self, predicted: torch.Tensor, target: torch.Tensor):
        batch_size = predicted.shape[0]

        predicted_per_image = []
        target_per_image = []

        for b in range(batch_size):
            current_predicted = predicted[b]
            current_target = target[b]

            # get mask of objects
            mask_obj_per_box = self.bb_ops.compute_confidence_mask(current_target)  # Shape: [grid_size, grid_size, num_bounding_boxes]
            mask_obj = mask_obj_per_box.any(dim=-1)  # Shape: [grid_size, grid_size]
            mask_box = mask_obj_per_box[mask_obj]  # Shape: [objects, num_boxes]

            # Get the bounding boxes
            corners_pred = self.bb_ops.to_corners(current_predicted[mask_obj])[mask_box]  # [num_objects, 4]
            corners_target = self.bb_ops.to_corners(current_target[mask_obj])[mask_box]  # [num_objects, 4]

            # Get the confidence
            confidence_pred = self.bb_ops.get_attr(current_predicted[mask_obj], 'confidence')[mask_box]  # Shape: [num_objects]

            # Get the classes
            class_pred = self.bb_ops.get_classes(current_predicted[mask_obj])  # [objects, num_classes]
            class_target = self.bb_ops.get_classes(current_target[mask_obj])  # [objects, num_classes]

            # Convert the class from one-hot to index
            class_pred = torch.argmax(class_pred, dim=-1)  # Shape: [num_objects]
            class_target = torch.argmax(class_target, dim=-1)  # Shape: [num_objects]

            # Repeat the class for each box
            class_pred = repeat(class_pred, 'c -> c n', n=self.num_bounding_boxes)[mask_box]  # Shape: [class, num_boxes]
            class_target = repeat(class_target, 'c -> c n', n=self.num_bounding_boxes)[mask_box]  # Shape: [class, num_boxes]

            # Create the expected list dicts
            predicted_output = {
                "boxes": corners_pred,
                "labels": class_pred,
                "scores": confidence_pred,
            }

            targets_output = {
                "boxes": corners_target,
                "labels": class_target,
            }

            predicted_per_image.append(predicted_output)
            target_per_image.append(targets_output)

        return predicted_per_image, target_per_image

    def compute(self):
        return self.mAP.compute()

    def __repr__(self):
        return self.mAP.__repr__()