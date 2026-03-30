"""
🎯 Real-Time Object Detection System
Built with YOLOv8 + Streamlit + OpenCV

Supports: Image Upload | Video Upload | Live Webcam
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

from detector import ObjectDetector
from utils import draw_detections, draw_fps, FPSCounter, format_detection_summary, resize_frame
from config import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_CONFIDENCE, DEFAULT_IOU, WEBCAM_INDEX, TARGET_DISPLAY_WIDTH


# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Object Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Main background ─────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Stat cards ──────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,199,255,0.12);
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.55) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00c7ff 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,199,255,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0,199,255,0.4);
}

/* ── File uploader ───────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(0,199,255,0.3);
    border-radius: 16px;
    padding: 20px;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,199,255,0.6);
}

/* ── Detection table ─────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Slider ──────────────────────────────────────────────────── */
div[data-testid="stSlider"] > div > div {
    color: #00c7ff;
}

/* ── Header gradient text ────────────────────────────────────── */
.gradient-text {
    background: linear-gradient(135deg, #00c7ff 0%, #7c3aed 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2.4rem;
    line-height: 1.2;
}
.subtitle {
    color: rgba(255,255,255,0.5);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: -8px;
}

/* ── Glass card ──────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* ── Badge ───────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-gpu { background: rgba(0,199,255,0.15); color: #00c7ff; }
.badge-cpu { background: rgba(255,149,0,0.15); color: #ff9500; }
.badge-active { background: rgba(52,199,89,0.15); color: #34c759; }

/* ── Divider ─────────────────────────────────────────────────── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,199,255,0.3), transparent);
    border: none;
    margin: 24px 0;
}

/* ── Hide streamlit branding ─────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached Model Loading ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str) -> ObjectDetector:
    """Load and cache the YOLOv8 model."""
    return ObjectDetector(model_path)


# ── Sidebar ──────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="gradient-text" style="font-size:1.6rem;">⚙️ Controls</p>', unsafe_allow_html=True)
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Model selection
        st.markdown("##### 🧠 Model")
        model_name = st.selectbox(
            "Select model variant",
            list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
            label_visibility="collapsed",
        )

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Thresholds
        st.markdown("##### 🎚️ Detection Thresholds")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05, max_value=1.0,
            value=DEFAULT_CONFIDENCE, step=0.05,
            help="Minimum confidence score to display a detection.",
        )
        iou = st.slider(
            "IOU Threshold (NMS)",
            min_value=0.05, max_value=1.0,
            value=DEFAULT_IOU, step=0.05,
            help="Intersection‑over‑Union threshold for non‑max suppression.",
        )

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Input source
        st.markdown("##### 📷 Input Source")
        source = st.radio(
            "Choose input",
            ["📸 Image Upload", "🎬 Video Upload", "📹 Live Webcam"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Model info
        model_path = AVAILABLE_MODELS[model_name]
        detector = load_model(model_path)
        device = detector.device
        badge = "badge-gpu" if "cuda" in device.lower() else "badge-cpu"
        device_label = "GPU" if "cuda" in device.lower() else "CPU"
        st.markdown(f"""
        <div class="glass-card" style="padding:16px;">
            <p style="margin:0;font-weight:600;color:rgba(255,255,255,0.8);">Model Info</p>
            <p style="margin:4px 0 0;font-size:0.85rem;color:rgba(255,255,255,0.45);">
                {model_name}<br>
                Device: <span class="badge {badge}">{device_label}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    return model_path, confidence, iou, source


# ── Header ───────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <p class="gradient-text">🎯 Real‑Time Object Detection</p>
        <p class="subtitle">Powered by YOLOv8 &nbsp;•&nbsp; Ultralytics &nbsp;•&nbsp; OpenCV</p>
    </div>
    <div class="styled-divider"></div>
    """, unsafe_allow_html=True)


# ── Detection Stats ─────────────────────────────────────────────────
def render_stats(detections: list[dict], processing_time: float = 0, fps: float = 0):
    summary = format_detection_summary(detections)
    total = len(detections)
    unique = len(summary)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Total Detections", total)
    col2.metric("🏷️ Unique Classes", unique)
    col3.metric("⚡ Processing Time", f"{processing_time:.0f} ms")
    col4.metric("🎞️ FPS", f"{fps:.1f}" if fps > 0 else "—")

    if summary:
        st.markdown("#### 📊 Detection Breakdown")
        breakdown_cols = st.columns(min(len(summary), 6))
        for i, (name, count) in enumerate(summary.items()):
            with breakdown_cols[i % len(breakdown_cols)]:
                st.metric(name.title(), count)


# ── Image Mode ───────────────────────────────────────────────────────
def run_image_mode(detector: ObjectDetector, confidence: float, iou: float):
    st.markdown("### 📸 Image Upload")
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("❌ Could not read the uploaded image.")
            return

        # Detect
        t0 = time.perf_counter()
        detections = detector.detect_image(image, confidence, iou)
        processing_time = (time.perf_counter() - t0) * 1000

        # Draw
        annotated = draw_detections(image.copy(), detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display side‑by‑side
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(original_rgb, use_container_width=True)
        with col_det:
            st.markdown("**Detected**")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Stats
        render_stats(detections, processing_time)

        # Detail table
        if detections:
            st.markdown("#### 📋 Detection Details")
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Class": d["class_name"].title(),
                    "Confidence": f'{d["confidence"]:.1%}',
                    "X1": int(d["box"][0]),
                    "Y1": int(d["box"][1]),
                    "X2": int(d["box"][2]),
                    "Y2": int(d["box"][3]),
                }
                for d in detections
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ── Video Mode ───────────────────────────────────────────────────────
def run_video_mode(detector: ObjectDetector, confidence: float, iou: float):
    st.markdown("### 🎬 Video Upload")
    uploaded = st.file_uploader(
        "Drop a video file here",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Write to temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Could not open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps_val = cap.get(cv2.CAP_PROP_FPS) or 30

        st.info(f"📽️ Video: {total_frames} frames @ {orig_fps_val:.0f} FPS")

        # Placeholders
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        stop_btn = st.button("⏹ Stop Processing", key="stop_video")

        fps_counter = FPSCounter()
        frame_idx = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame, TARGET_DISPLAY_WIDTH)

            # Detect
            t0 = time.perf_counter()
            detections = detector.detect(frame, confidence, iou)
            proc_ms = (time.perf_counter() - t0) * 1000

            # Draw
            annotated = draw_detections(frame.copy(), detections)
            fps_val = fps_counter.tick()
            annotated = draw_fps(annotated, fps_val)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(annotated_rgb, use_container_width=True)

            with stats_placeholder.container():
                render_stats(detections, proc_ms, fps_val)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

        cap.release()
        progress_bar.progress(1.0)
        st.success("✅ Video processing complete!")


# ── Webcam Mode ──────────────────────────────────────────────────────
def run_webcam_mode(detector: ObjectDetector, confidence: float, iou: float):
    st.markdown("### 📹 Live Webcam")
    st.markdown("""
    <div class="glass-card" style="padding:16px;">
        <p style="margin:0;color:rgba(255,255,255,0.6);font-size:0.9rem;">
            🔴 Press <strong>Start</strong> to begin live detection. Press <strong>Stop</strong> to end.
            Make sure your webcam is connected and accessible.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_start, col_stop = st.columns(2)
    start = col_start.button("▶️ Start Webcam", key="start_webcam", use_container_width=True)
    stop = col_stop.button("⏹ Stop Webcam", key="stop_webcam", use_container_width=True)

    if start:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Check that it is connected and not used by another app.")
            return

        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        fps_counter = FPSCounter()

        st.session_state["webcam_running"] = True

        while st.session_state.get("webcam_running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame from webcam.")
                break

            frame = resize_frame(frame, TARGET_DISPLAY_WIDTH)

            # Detect
            t0 = time.perf_counter()
            detections = detector.detect(frame, confidence, iou)
            proc_ms = (time.perf_counter() - t0) * 1000

            # Draw
            annotated = draw_detections(frame.copy(), detections)
            fps_val = fps_counter.tick()
            annotated = draw_fps(annotated, fps_val)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(annotated_rgb, use_container_width=True)

            with stats_placeholder.container():
                render_stats(detections, proc_ms, fps_val)

        cap.release()

    if stop:
        st.session_state["webcam_running"] = False
        st.info("📹 Webcam stopped.")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    render_header()
    model_path, confidence, iou, source = render_sidebar()
    detector = load_model(model_path)

    if "📸" in source:
        run_image_mode(detector, confidence, iou)
    elif "🎬" in source:
        run_video_mode(detector, confidence, iou)
    elif "📹" in source:
        run_webcam_mode(detector, confidence, iou)


if __name__ == "__main__":
    main()
