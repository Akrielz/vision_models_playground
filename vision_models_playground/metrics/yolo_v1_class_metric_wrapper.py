import torch
import torchmetrics

from vision_models_playground.data_structures.yolo_bounding_box import YoloBoundingBoxOperations


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
        self.bb_ops = YoloBoundingBoxOperations(
            num_bounding_boxes=num_bounding_boxes,
            num_classes=num_classes
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        class_pred = self.bb_ops.get_classes(preds)
        class_target = self.bb_ops.get_classes(target)

        mask_obj_per_box = self.bb_ops.compute_confidence_mask(target)
        mask_obj = mask_obj_per_box.any(dim=-1)

        self.metric.update(class_pred[mask_obj], class_target[mask_obj])

    def compute(self):
        return self.metric.compute()

    def __repr__(self):
        return self.metric.__repr__()