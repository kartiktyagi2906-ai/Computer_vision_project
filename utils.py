"""
Utility helpers for drawing, FPS calculation, and result formatting.
"""

import time
import cv2
import numpy as np
from config import get_color_for_class, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS, LABEL_PADDING


# ── FPS Counter ──────────────────────────────────────────────────────

class FPSCounter:
    """Smoothed FPS counter using a rolling window."""

    def __init__(self, window: int = 30):
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Drawing Helpers ──────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores on a frame.

    Each detection dict: {class_id, class_name, confidence, box: [x1,y1,x2,y2]}
    """
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = get_color_for_class(det["class_id"])
        label = f'{det["class_name"]} {det["confidence"]:.0%}'

        # Semi‑transparent filled box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        label_y1 = max(y1 - th - LABEL_PADDING * 2, 0)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + LABEL_PADDING * 2, label_y2), color, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x1 + LABEL_PADDING, label_y2 - LABEL_PADDING // 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # Blend overlay (semi‑transparent fill inside boxes)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw an FPS badge in the top‑right corner."""
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    h, w = frame.shape[:2]
    x = w - tw - 20
    y = 10
    cv2.rectangle(frame, (x - 10, y), (x + tw + 10, y + th + 16), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 10, y), (x + tw + 10, y + th + 16), (0, 255, 163), 2)
    cv2.putText(frame, text, (x, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 163), 2, cv2.LINE_AA)
    return frame


# ── Result Formatting ────────────────────────────────────────────────

def format_detection_summary(detections: list[dict]) -> dict[str, int]:
    """Return a {class_name: count} summary."""
    counts: dict[str, int] = {}
    for det in detections:
        name = det["class_name"]
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def resize_frame(frame: np.ndarray, target_width: int = 720) -> np.ndarray:
    """Resize keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    ratio = target_width / w
    new_size = (target_width, int(h * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
