"""
Object Detection Engine — wraps Ultralytics YOLOv8.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """High-level wrapper around a YOLOv8 model."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model_path = model_path

    # ── Core detection ───────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = 0.40,
        iou: float = 0.45,
    ) -> list[dict]:
        """
        Run detection on a single frame (BGR numpy array).

        Returns a list of dicts:
            {class_id, class_name, confidence, box: [x1, y1, x2, y2]}
        """
        results = self.model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                })
        return detections

    # ── Convenience: detect on a PIL / file ──────────────────────────

    def detect_image(self, image: np.ndarray, confidence: float = 0.40, iou: float = 0.45):
        """Detect objects in a single image (as BGR numpy array)."""
        return self.detect(image, confidence, iou)

    # ── Model info ───────────────────────────────────────────────────

    @property
    def class_names(self) -> dict:
        return self.model.names

    @property
    def device(self) -> str:
        """Return the device the model is running on."""
        try:
            return str(self.model.device)
        except Exception:
            return "cpu"
