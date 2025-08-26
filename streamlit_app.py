import os
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from mole_analysis_pipeline import MoleAnalysisPipeline
from full_body_pipeline import FullBodyMoleAnalysisPipeline

# ----------------------------
# Helper functions & caching
# ----------------------------

@st.cache_resource(show_spinner=False)
def _load_single_pipeline():
    """Load single-mole analysis pipeline once and cache it."""
    return MoleAnalysisPipeline()


@st.cache_resource(show_spinner=False)
def _load_full_pipeline():
    """Load full-body analysis pipeline once and cache it."""
    return FullBodyMoleAnalysisPipeline(
        yolo_model_path='weights/best_1280_default_hyper.pt',
        segmentation_model_path='weights/segment_mob_unet_.bin'
    )


def _save_temp_image(uploaded_file) -> Path:
    """Persist uploaded in-memory file to a NamedTemporaryFile and return its path."""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    return Path(tmp.name)


def _styled_metric(key: str, value: float):
    """Display ABCD metric using Streamlit metric component with dynamic coloring."""
    colors = {
        'Asymmetry': '#e74c3c',
        'Border': '#f1c40f',
        'Colour': '#8e44ad',
        'Diameter': '#2980b9'
    }
    color = colors.get(key, '#2ecc71')
    st.markdown(f"""
        <div style='padding:8px;border-radius:8px;background:{color};text-align:center'>
            <span style='font-weight:600;color:white'>{key}</span><br>
            <span style='font-size:24px;font-weight:700;color:white'>{value:.2f}</span>
        </div>
    """, unsafe_allow_html=True)




def _draw_mole_highlight(image, bbox, color=(0, 255, 0), thickness=3):
    """Return image copy with a rectangle drawn around the mole bbox.
    bbox can be [x, y, w, h] in absolute pixels or normalized floats.
    Safe-guards against malformed bboxes."""
    if not bbox or len(bbox) != 4:
        return image
    x1, y1, x2, y2 = bbox
    # If bbox is x,y,w,h convert to x1,y1,x2,y2
    if x2 <= x1 or y2 <= y1:  # assume given as width/height
        x2 = x1 + x2
        y2 = y1 + y2
    # Normalize if values in 0-1
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        h_img, w_img = image.shape[:2]
        x1 = int(x1 * w_img)
        y1 = int(y1 * h_img)
        x2 = int(x2 * w_img)
        y2 = int(y2 * h_img)
    try:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    except Exception:
        return image
    # Clamp
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(x1, w_img-1))
    y1 = max(0, min(y1, h_img-1))
    x2 = max(x1+1, min(x2, w_img-1))
    y2 = max(y1+1, min(y2, h_img-1))
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# ----------------------------
# UI LAYOUT
# ----------------------------

st.set_page_config(
    page_title="DermAI ‑ Mole Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("DermAI")
mode = st.sidebar.radio("Select Mode", ["Single Mole", "Full Body"])

st.title("DermAI – Intelligent Skin Lesion Analysis")

with st.sidebar.expander("About", expanded=False):
    st.write("Upload a skin image and instantly receive dermatological insights using state-of-the-art segmentation and ABCD analysis models.")
    #st.markdown("Built with **PyTorch**, **Streamlit**, and ❤.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if uploaded:
    image_path = _save_temp_image(uploaded)
    original_image = cv2.imread(str(image_path))
    if 'selected_mole' not in st.session_state:
        st.session_state.selected_mole = None
    col1, col2, col3 = st.columns([2, 1, 1])
    # with col1:
    #     st.image(str(image_path), caption="Original", use_container_width=True)

    if mode == "Single Mole":
        pipeline = _load_single_pipeline()
        with st.spinner("Analyzing image …"):

            results = pipeline.process_image(str(image_path), save_outputs=True)
            mask = results["mask"]
            image_path = results["upscaled_image"]
            overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            original = cv2.imread(str(image_path))
            original = cv2.resize(original, (mask.shape[1], mask.shape[0]))
            blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
            metrics = results["metrics"]
            
        with col1:
            st.image(str(image_path), caption="Original", use_container_width=True)

        with col2:
            st.image(blended[..., ::-1], caption="Segmentation Overlay", use_container_width=True)

        st.subheader("ABCD Metrics")
        metric_cols = st.columns(4)
        for (m, v), c in zip(metrics.items(), metric_cols):
            with c:
                if isinstance(v, dict):
                    st.write(m)
                    st.json(v)
                else:
                    _styled_metric(m, v)

        st.download_button("Download Metrics JSON", json.dumps(metrics, indent=2), file_name="metrics.json", mime="application/json")

    else:  # Full Body mode
        pipeline = _load_full_pipeline()
        with st.spinner("Analyzing full-body image …"):
            fb_results = pipeline.process_full_body_image(str(image_path), output_dir=str(image_path.parent))

        # fb_results may be dict (new) or list (legacy). Standardize.
        if isinstance(fb_results, list):
            mole_crops = [cv2.imread(m.get('cropped_image_path', '')) for m in fb_results]
            mole_metrics = [m.get('analysis', {}) for m in fb_results]
            mole_bboxes = [m.get('bbox', []) for m in fb_results]
        else:
            mole_crops = fb_results.get('mole_crops', [])
            mole_metrics = fb_results.get('metrics', [])
            mole_bboxes = fb_results.get('mole_bboxes', [])

        # --- c1 : full image ---
        with col1:
            st.subheader("Full Body Scan")
            display_img = original_image
            if st.session_state.selected_mole is not None and st.session_state.selected_mole < len(mole_bboxes):
                display_img = _draw_mole_highlight(original_image, mole_bboxes[st.session_state.selected_mole])
            st.image(display_img, channels="BGR", use_container_width=True)

        # --- c3 : metrics display ---
        with col3:
            st.subheader("Selected Metrics")
            if st.session_state.selected_mole is not None and st.session_state.selected_mole < len(mole_metrics):
                sel_met = mole_metrics[st.session_state.selected_mole]
                if sel_met:
                    for k, v in sel_met.items():
                        # Skip raw detailed metrics to avoid clutter
                        if k == "Raw_Metrics":
                            continue
                        if isinstance(v, dict):
                            st.write(k)
                            st.json(v)
                        else:
                            _styled_metric(k, v)
            else:
                st.write("Select a crop to view its metrics.")

        # --- c2 : crop list ---
        with col2:
            st.subheader(f"Detected Moles ({len(mole_crops)})")

            # Add custom CSS for a fixed-height scrollable area
            st.markdown(
                """
                <style>
                /* Limit second column (crop list) vertical block height and enable scrolling */
                div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] {
                    max-height: 500px;
                    overflow-y: auto;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Streamlit components rendered inside this container will stack vertically
            with st.container():
                for idx, (crop, met) in enumerate(zip(mole_crops, mole_metrics)):
                    if st.button(f"Select", key=f"mole_{idx}"):
                        st.session_state.selected_mole = idx
                        st.rerun()
                    # Display crop with correct color (BGR)
                    st.image(crop, channels="BGR", use_container_width=True)
                
    # Clean up temp image when session ends
    st.session_state.setdefault("_cleanup", []).append(str(image_path))


def _session_cleanup():
    for p in st.session_state.get("_cleanup", []):
        try:
            os.remove(p)
        except OSError:
            pass


import atexit
atexit.register(_session_cleanup)
