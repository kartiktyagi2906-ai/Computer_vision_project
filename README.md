#  Real-Time Object Detection System

A real-time object detection system built with **YOLOv8** and **Streamlit** that can detect and classify 80+ object types from images, videos, or a live webcam feed.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)

---

##  Features

- **Three Input Modes**: Image upload, video upload, and live webcam feed
- **Multiple Model Variants**: Switch between YOLOv8 Nano (fast), Small (balanced), and Medium (accurate)
- **Real-Time Performance**: Optimized for ~15–30+ FPS on modern hardware
- **Interactive Dashboard**: Live detection stats, object counts, FPS monitoring
- **Adjustable Thresholds**: Confidence and IOU sliders for fine-tuned control
- **Premium Dark UI**: Glassmorphism design with smooth animations
- **80 Object Classes**: Detects people, vehicles, animals, everyday items, and more (COCO dataset)

---

##  Setup & Installation

### 1. Prerequisites
- Python 3.8 or later
- pip (Python package manager)
- Webcam (optional, for live detection)

### 2. Install Dependencies

```bash
cd cv_project
pip install -r requirements.txt
```

> **Note**: The first run will automatically download the YOLOv8 model weights (~6MB for Nano).

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

##  Usage

###  Image Detection
1. Select **Image Upload** in the sidebar
2. Upload a JPG/PNG image
3. View the annotated result with bounding boxes and confidence scores

###  Video Detection
1. Select ** Video Upload** in the sidebar
2. Upload an MP4/AVI/MOV video
3. Watch real-time frame-by-frame detection with FPS overlay

###  Live Webcam
1. Select ** Live Webcam** in the sidebar
2. Click ** Start Webcam**
3. See live detection with bounding boxes and performance metrics

###  Controls
- **Model Selection**: Choose between Nano (fastest), Small (balanced), or Medium (most accurate)
- **Confidence Threshold**: Filter out low-confidence detections (default: 0.40)
- **IOU Threshold**: Adjust non-max suppression overlap tolerance (default: 0.45)

---

##  Project Structure

```
cv_project/
├── app.py              # Main Streamlit application
├── detector.py         # YOLOv8 detection engine
├── utils.py            # Drawing utilities & FPS counter
├── config.py           # Configuration & constants
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit theme
```

---

##  How It Works

1. **Model Loading**: A pretrained YOLOv8 model is loaded via Ultralytics
2. **Input Processing**: Images/video frames are read via OpenCV
3. **Object Detection**: YOLOv8 predicts bounding boxes, class labels, and confidence scores
4. **Visualization**: Results are drawn on frames with color-coded bounding boxes
5. **Dashboard**: Statistics are computed and displayed in real-time

---

##  Technologies

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **YOLOv8 (Ultralytics)** | Object detection model |
| **Streamlit** | Web application framework |
| **OpenCV** | Image/video processing |
| **PyTorch** | Deep learning backend (via Ultralytics) |

---

##  Detected Object Classes (COCO)

The system can detect **80 classes** including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Everyday Items**: backpack, umbrella, handbag, suitcase, bottle, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Furniture**: chair, couch, bed, dining table
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- And many more!

---

##  License

This project is for educational purposes.
