from typing import Tuple, List, Dict, Optional, Any

import torch
from PIL.Image import Image
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor

from vision_models_playground.data_structures.yolo_bounding_box import YoloBoundingBoxOperations
from vision_models_playground.datasets.datasets import get_voc_detection_dataset_raw, get_voc_detection_dataset_yolo
from vision_models_playground.predictors.base_predictor import Predictor
from vision_models_playground.utility.load_models import load_best_model


YoloObjects = List[Dict[str, Any]]


class YoloV1Predictor(Predictor):
    def __init__(
            self,
            model: nn.Module,
            intermediate_size: Tuple[int, int] = (448, 448),
            threshold: float = 0.02,
            num_bounding_boxes: int = 2,
            num_classes: int = 20,
            grid_size: int = 7,
            class_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__(model, self.collate_in, self.collate_out)

        self.transform_in = Compose([
            ToTensor(),
            Resize(intermediate_size),
        ])

        self.threshold = threshold
        self.grid_size = grid_size
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes

        self.bb_ops = YoloBoundingBoxOperations(
            num_bounding_boxes=num_bounding_boxes,
            num_classes=num_classes
        )

        self.class_to_idx_map = class_map
        self.idx_to_class_map = {v: k for k, v in self.class_to_idx_map.items()} if class_map is not None else None

    def collate_in(self, images: List[Image] | Image):
        if not isinstance(images, list):
            images = [images]

        images_transformed = [self.transform_in(image) for image in images]
        images_transformed = torch.stack(images_transformed)
        return images_transformed

    def collate_out(self, orig_images: List[Image] | Image, output: torch.Tensor):
        if not isinstance(orig_images, list):
            orig_images = [orig_images]

        # Get the confidence for each bounding box
        confidences = self.bb_ops.get_attr(output, 'confidence')

        # Build the masks
        mask_per_bbox = confidences > self.threshold  # (batch_size, grid_size, grid_size, num_bounding_boxes)

        # Get the rows and columns of the bounding boxes
        batch, rows, cols, boxes = torch.where(mask_per_bbox)

        # Get the classes
        class_prob = self.bb_ops.get_classes(output)
        class_ids = torch.argmax(class_prob, dim=-1)

        # Get the bounding boxes
        corners = self.bb_ops.to_corners_absolute(output)

        # Each image will have a list of objects
        # Each object will have a dict defining the class, confidence and corners
        output_list: List[YoloObjects] = [[] for _ in orig_images]

        # Keep track of objects already seen
        seen = set()
        for (i, r, c, b) in zip(batch, rows, cols, boxes):
            if (i, r, c) in seen:
                continue

            class_id = class_ids[i, r, c].cpu().item()
            class_name = self.idx_to_class_map[class_id] if self.idx_to_class_map is not None else None

            confidence = confidences[i, r, c, b].cpu().item()
            bbox = corners[i, r, c, b].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox

            orig_size = orig_images[i].size
            width, height = orig_size
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            obj_dict = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            }

            output_list[i].append(obj_dict)
            seen.add((i, r, c))

        return output_list


def main():
    model = load_best_model("models/train/ResNetYoloV1/2023-07-06_14-37-23")
    class_map = get_voc_detection_dataset_yolo()[1].class_map
    predictor = YoloV1Predictor(model, threshold=0.2, class_map=class_map)
    voc_test = get_voc_detection_dataset_raw()[1]
    image = voc_test[0][0]
    predicted = predictor.predict(image)
    print(predicted)


if __name__ == '__main__':
    main()