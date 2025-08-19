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
    col1, col2 = st.columns(2)
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
        with st.spinner("Detecting and analyzing lesions …"):
            results = pipeline.process_full_body_image(str(image_path), output_dir=str(image_path.parent))
        st.subheader(f"Detected {len(results)} lesion(s)")
        for idx, res in enumerate(results, 1):
            st.markdown(f"### Lesion {idx}")
            c1, c2 = st.columns([2, 4])
            with c1:
                st.image(res.get("cropped_image_path", str(image_path)), caption="Lesion Crop", width=200)
            with c2:
                analysis = res.get("analysis", {})
                if not analysis:
                    st.write("No metrics")
                elif "error" in analysis:
                    st.error(analysis["error"])
                else:
                    for k, v in analysis.items():
                        if isinstance(v, dict):
                            if k != "Raw_Metrics":
                                st.write(k)
                                st.json(v)
                        else:
                            _styled_metric(k, v)

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
