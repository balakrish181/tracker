"""
Shared constants for the mole analysis pipeline.
Centralizes values that were previously hardcoded across multiple files.
"""

# ABCD metric keys — used in compare_pipeline, compare_loftr_pipeline, and analysis output
ABCD_KEYS = ["Asymmetry", "Border", "Colour", "Diameter"]

# Default model weight paths
YOLO_WEIGHTS = "weights/best_1280_default_hyper.pt"
SEGMENTATION_WEIGHTS = "weights/segment_mob_unet_.bin"
ESRGAN_WEIGHTS = "weights/dermaRealESRGAN_x2plus_v1.pth"
LOFTR_WEIGHTS = "weights/outdoor_ds.ckpt"

# Image validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_UPLOAD_SIZE_MB = 50
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 20000

# YOLO detection
MIN_DETECTION_CONFIDENCE = 0.25
MIN_DETECTION_SIZE_PX = 8


def safe_get_metric(d: dict, key: str):
    """Safely extract a numeric metric value from a results dict."""
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def percent_change(v1, v2):
    """Calculate percentage change between two metric values. Returns None if undefined."""
    if v1 is None or v2 is None:
        return None
    if v1 == 0:
        return None
    return ((v2 - v1) / v1) * 100.0
