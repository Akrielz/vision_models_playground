from pprint import pprint
from typing import Tuple, List, Dict, Optional, Any

import PIL
import cv2
import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor

from vision_models_playground.data_structures.yolo_bounding_box import YoloBoundingBoxOperations
from vision_models_playground.datasets.datasets import get_voc_detection_dataset_raw
from vision_models_playground.pipelines.base import Pipeline
from vision_models_playground.utility.hub import load_vmp_pipeline_from_hub

YoloObjects = Dict[str, Any]


class YoloV1Pipeline(Pipeline):
    def __init__(
            self,
            model: Optional[nn.Module] = None,
            *,
            intermediate_size: Tuple[int, int] = (448, 448),
            threshold: float = 0.20,
            max_overlap: float = 0.25,
            num_bounding_boxes: int = 2,
            num_classes: int = 20,
            grid_size: int = 7,
            class_map: Optional[Dict[str, int]] = None,
            device: Optional[torch.device] = None
    ):
        super().__init__(model, self.collate_in, self.collate_out, device=device)

        self.transform_in = Compose([
            ToTensor(),
            Resize(intermediate_size, antialias=True),
        ])

        if class_map is not None:
            num_classes = len(class_map)

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

        self.max_overlap = max_overlap

    def collate_in(self, images: List[Image] | Image):
        if not isinstance(images, list):
            images = [images]

        images_transformed = [self.transform_in(image) for image in images]
        images_transformed = torch.stack(images_transformed)
        images_transformed = images_transformed.to(self.device)
        return images_transformed

    def _prepare_indices(self, confidences: torch.Tensor, num_images: int):
        mask_per_bbox = confidences > self.threshold  # (batch_size, grid_size, grid_size, num_bounding_boxes)

        batch, rows, cols, boxes = [], [], [], []
        for i in range(num_images):
            r, c, b = torch.where(mask_per_bbox[i])

            # Sort the bounding boxes by confidence
            image_confidences = confidences[i, r, c, b]
            _, sorted_indices = torch.sort(image_confidences, descending=True)

            rows.append(r[sorted_indices])
            cols.append(c[sorted_indices])
            boxes.append(b[sorted_indices])
            batch.append(torch.full_like(r, i))

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        boxes = torch.cat(boxes)
        batch = torch.cat(batch)

        return batch, rows, cols, boxes

    def _compute_output_objects(
            self,
            orig_images: List[Image],
            batch: torch.Tensor,
            rows: torch.Tensor,
            cols: torch.Tensor,
            boxes: torch.Tensor,
            confidences: torch.Tensor,
            class_ids: torch.Tensor,
            corners: torch.Tensor
    ):
        output_list: List[YoloObjects] = [{"objects": [], "image": None} for _ in orig_images]

        occupied_area = [
            torch.zeros(image.size[1], image.size[0], device=self.device, dtype=torch.bool)
            for image in orig_images
        ]

        for (i, r, c, b) in zip(batch, rows, cols, boxes):
            class_id = class_ids[i, r, c].cpu().item()
            class_name = self.idx_to_class_map[class_id] if self.idx_to_class_map is not None else None

            confidence = confidences[i, r, c, b].cpu().item()
            bbox = corners[i, r, c, b].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox

            orig_size = orig_images[i].size
            width, height = orig_size
            x_min = int(np.clip(x_min * width - 1, a_min=0, a_max=width - 1))
            y_min = int(np.clip(y_min * height - 1, a_min=0, a_max=height - 1))
            x_max = int(np.clip(x_max * width + 1, a_min=0, a_max=width - 1))
            y_max = int(np.clip(y_max * height + 1, a_min=0, a_max=height - 1))

            bbox_area = (x_max - x_min) * (y_max - y_min)
            already_occupied = occupied_area[i][y_min:y_max, x_min:x_max].sum().cpu().item()
            if already_occupied / bbox_area > self.max_overlap:
                continue

            occupied_area[i][y_min:y_max, x_min:x_max] = True

            obj_dict = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'overlap': already_occupied / bbox_area,
                'area': bbox_area,
            }

            output_list[i]['objects'].append(obj_dict)

        return output_list

    def _draw_objects_on_image(self, image: Image, objects: List[Dict[str, Any]]):
        image_edited = np.array(image)
        for obj in objects:
            # assign a random color
            color = hash(obj['class_name'])
            color = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF)

            x_min = obj['x_min']
            y_min = obj['y_min']
            x_max = obj['x_max']
            y_max = obj['y_max']
            name = obj['class_name']
            confidence = obj['confidence']

            display = f"{name} {confidence:.2f}"

            cv2.rectangle(image_edited, (x_min, y_min), (x_max, y_max), color, 2)
            text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

            # Determine how big the text_width is compared to the bounding box
            text_width = text_size[0]
            text_height = text_size[1]
            box_width = x_max - x_min
            box_height = y_max - y_min

            # If the text is wider than the bounding box, then we need to compute the scale factor
            scale_factor = 1.0
            if text_width > box_width:
                scale_factor *= box_width / text_width
                text_width = int(scale_factor * text_width)
                text_height = int(scale_factor * text_height)

            if text_height > box_height:
                scale_factor *= box_height / text_height
                text_width = int(scale_factor * text_width)
                text_height = int(scale_factor * text_height)

            cv2.putText(image_edited, display, (x_min, y_min + text_height), cv2.FONT_HERSHEY_SIMPLEX, scale_factor, color, 2)

        image_edited = PIL.Image.fromarray(image_edited)
        return image_edited

    def collate_out(self, orig_images: List[Image] | Image, output: torch.Tensor) -> List[YoloObjects]:
        if not isinstance(orig_images, list):
            orig_images = [orig_images]

        # Get the confidence for each bounding box
        confidences = self.bb_ops.get_attr(output, 'confidence')

        # Get the rows and columns of the bounding boxes
        batch, rows, cols, boxes = self._prepare_indices(confidences, len(orig_images))

        # Get the classes
        class_prob = self.bb_ops.get_classes(output)
        class_ids = torch.argmax(class_prob, dim=-1)

        # Get the bounding boxes
        corners = self.bb_ops.to_corners_absolute(output)

        # Compute the objects
        objects = self._compute_output_objects(
            orig_images=orig_images,
            batch=batch,
            rows=rows,
            cols=cols,
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            corners=corners
        )

        # Prepare an image with the bounding boxes
        bbox_images = [
            self._draw_objects_on_image(image, objects[i]['objects'])
            for i, image in enumerate(orig_images)
        ]

        for i, image in enumerate(bbox_images):
            objects[i]['image'] = image

        return objects


def main():
    pipeline = load_vmp_pipeline_from_hub('Akriel/ResNetYoloV1')

    voc_test = get_voc_detection_dataset_raw()[1]
    image = voc_test[9][0]
    predicted = pipeline(image)
    pprint(predicted)


if __name__ == '__main__':
    main()