import base64
import io
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image


def load_image_to_path(image_b64: Optional[str] = None, image_url: Optional[str] = None, suffix: str = ".jpg") -> Tuple[str, Tuple[int, int]]:
    """Load image from base64 or URL into a temp file under /tmp.

    Returns (path, (width, height)).
    """
    if not image_b64 and not image_url:
        raise ValueError("Provide either image_b64 or image_url")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(tmp_fd)

    if image_b64:
        data = base64.b64decode(image_b64)
        with open(tmp_path, "wb") as f:
            f.write(data)
    else:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

    img = cv2.imread(tmp_path)
    if img is None:
        raise ValueError("Failed to decode image")
    h, w = img.shape[:2]
    return tmp_path, (w, h)


def image_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
