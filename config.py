"""
Configuration & Constants for the Object Detection System.
"""

# ── Model Configuration ──────────────────────────────────────────────
AVAILABLE_MODELS = {
    "YOLOv8 Nano (Fastest)": "yolov8n.pt",
    "YOLOv8 Small (Balanced)": "yolov8s.pt",
    "YOLOv8 Medium (Accurate)": "yolov8m.pt",
}

DEFAULT_MODEL = "YOLOv8 Nano (Fastest)"

# ── Detection Thresholds ─────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.45
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 1.0
MIN_IOU = 0.05
MAX_IOU = 1.0

# ── Display Settings ─────────────────────────────────────────────────
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_PADDING = 8

# ── Video / Webcam Settings ─────────────────────────────────────────
WEBCAM_INDEX = 0
MAX_VIDEO_FRAMES = 10000       # safety limit
TARGET_DISPLAY_WIDTH = 720

# ── Color Palette (BGR for OpenCV) ───────────────────────────────────
# A vibrant, high-contrast palette for bounding boxes
CLASS_COLORS = [
    (255, 76, 76),    # coral-red
    (76, 217, 100),   # green
    (0, 199, 255),    # amber/gold
    (255, 149, 0),    # orange (BGR)
    (88, 86, 214),    # purple
    (255, 45, 85),    # pink
    (90, 200, 250),   # teal
    (0, 122, 255),    # blue
    (175, 82, 222),   # violet
    (52, 199, 89),    # mint
    (255, 204, 0),    # yellow
    (162, 132, 94),   # brown
    (142, 142, 147),  # gray
    (0, 255, 163),    # spring-green
    (255, 59, 48),    # red
    (50, 173, 230),   # sky-blue
    (255, 179, 64),   # tangerine
    (191, 90, 242),   # magenta
    (48, 176, 199),   # cyan
    (99, 230, 226),   # turquoise
]

# ── COCO Class Names (80 classes) ────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_color_for_class(class_id: int) -> tuple:
    """Return a consistent BGR color for a given class ID."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]
