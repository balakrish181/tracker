# Mole Analysis App (v2) — Architecture Overview

This document explains the **end-to-end logic** of the software (from user upload to final results), and the **tools/tech stack** used and **why** they were chosen.

---

## What the software does (big picture)

The app supports four related workflows:

- **Single mole analysis**: Upload a single mole image → segment lesion → compute **ABCD** metrics → return metrics + mask/overlay images.
- **Full-body screening**: Upload a full-body image → detect all moles → crop each mole → run the same single-mole pipeline on each crop → return list of detected moles + metrics.
- **Single-to-single comparison**: Upload two single-mole images → analyze both → compute **percent change** per ABCD metric.
- **Full-body to full-body comparison (LoFTR)**: Upload two full-body images → detect moles in both → match corresponding moles across time using **LoFTR** + YOLO centers → analyze each paired crop → compute per-pair percent change.

---

## Where the main logic lives (files)

### Backend (Flask + CV/ML pipelines)

- `app.py`
  - Flask server
  - Routes (pages + JSON APIs)
  - Upload/output folder handling
  - Instantiates pipelines:
    - `IntegratedMolePipeline`
    - `FullBodyMoleAnalysisPipeline`
    - `CompareMolePipeline`
    - `LoFTRFullBodyComparator`

- `integrated_pipeline.py`
  - The **core single-image pipeline**: segmentation → ABCD analysis → optional overlay rendering.

- `full_body_pipeline.py`
  - Full-body workflow: YOLO detect → crop → (optional resize/pad) → **Real-ESRGAN upscaling** → per-crop integrated pipeline.

- `compare_pipeline.py`
  - Single-to-single comparison: run integrated pipeline on both → percent change.

- `compare_loftr_pipeline.py`
  - Full-body comparison:
    - YOLO detect both images
    - Pair detections using LoFTR matches around YOLO box centers
    - Crop matched moles
    - Run integrated pipeline on each crop
    - Percent change per pair

### Frontend (templates/UI)

- `templates/base.html`
  - Shared layout (header/nav), Tailwind import, small shared UI helpers (toast/loading).
- `templates/dashboard.html`
  - Landing “dashboard” page with links into each workflow.
- `templates/single.html`, `templates/full_body.html`, `templates/compare.html`, `templates/compare_full_body.html`
  - Page templates for each workflow.
- `templates/index.html`
  - A self-contained Tailwind + JS page that directly calls endpoints like `/analyze`, `/analyze_full_body`, `/compare`, `/compare_full_body`.

### Metrics / analysis

- `metrics/merged_improved_metrics.py`
  - The primary analyzer used by the integrated pipeline (`MoleAnalyzer`).
- `metrics/improved_metrics.py` (and related modules)
  - Shows representative logic for ABCD calculations (asymmetry, border irregularity, diameter, colour variation).

### Model components / helpers (examples)

- `seg_mole_metrics/inference.py`
  - Segmentation inference wrapper (`MobileUNETRInference`).
- `realesrgan_upscaler.py` + `Real_ESRGAN/`
  - Super-resolution upscaler used on mole crops.
- `loftr/loftr_matcher.py` + `loftr/match_dir/`
  - LoFTR matcher logic and dependencies.

---

## End-to-end: from user action to final result

The UI sends images to Flask endpoints; the backend saves images, runs the appropriate pipeline, and returns JSON that the UI renders.

### 1) Single mole analysis (`POST /analyze`)

**Goal**: compute ABCD metrics on one mole image.

**Backend steps** (implemented in `app.py` and `integrated_pipeline.py`):

1. **Upload & save**
   - `app.py` saves the uploaded image to `uploads/` with a timestamped filename.
2. **Segmentation**
   - `IntegratedMolePipeline.process_image()` calls:
     - `MobileUNETRInference.predict(image_path)` to generate a **binary mask** (lesion vs background).
   - Mask is written to disk as `<stem>_mask.png`.
3. **ABCD analysis**
   - `MoleAnalyzer(image_path, mask_path).analyze(show=False)` returns a dict with keys like:
     - `Asymmetry`, `Border`, `Colour`, `Diameter` (plus optional raw metrics)
4. **Visualization assets**
   - Overlay is built using OpenCV blending and saved as `<stem>_overlay.png`.
5. **Response**
   - JSON includes:
     - `metrics` dict
     - URLs for original image, mask, overlay for UI display

**Frontend behavior**:

- Sends a `multipart/form-data` upload to `/analyze`.
- Renders returned metrics and images (original/mask/overlay).

---

### 2) Full-body screening (`POST /analyze_full_body`)

**Goal**: detect many moles in a full-body image, crop each, run the same ABCD analysis per mole.

**Backend steps** (`full_body_pipeline.py`):

1. **Mole detection (YOLOv5)**
   - Runs a custom YOLOv5 detector loaded via `torch.hub`.
   - For very large images, it:
     - splits into overlapping patches,
     - runs YOLO per patch,
     - remaps patch detections to full-image coordinates,
     - runs **non-maximum suppression** to remove duplicates.
2. **Crop each detection**
   - For each YOLO box, crop the mole region.
   - Optionally normalize crop size (e.g., pad/resize) for consistency.
3. **Enhance crop (Real-ESRGAN)**
   - Upscales each crop via `DermaRealESRGANx2` to improve detail.
4. **Run single-mole pipeline per crop**
   - Calls `IntegratedMolePipeline.process_image()` on each cropped mole.
5. **Response**
   - JSON contains:
     - `results`: list of objects like:
       - `mole_id`
       - `bbox` (normalized coords in the original full-body image)
       - `cropped_image_path`
       - `analysis` (ABCD metrics)
     - `original_image`
     - `image_dimensions`

**Frontend behavior**:

- Draws bounding boxes on the original full-body image using normalized bbox coordinates.
- Displays a grid of mole cards.
- Clicking a bbox/card shows metrics and the cropped image.

---

### 3) Single-to-single comparison (`POST /compare`)

**Goal**: compute ABCD on two mole images and show the percent change.

**Backend steps** (`compare_pipeline.py`):

1. Analyze image 1 using `IntegratedMolePipeline.process_image()`
2. Analyze image 2 using `IntegratedMolePipeline.process_image()`
3. Compute percent change per metric:
   - \[
     \%\Delta = \frac{(v_2 - v_1)}{v_1} \times 100
     \]
   - If `v1` is missing or 0, percent change is returned as `None`.
4. Return:
   - `image1_metrics`, `image2_metrics`, `percent_change`

**Frontend behavior**:

- Displays both images, their metrics, and % change cards.

---

### 4) Full-body to full-body comparison (`POST /compare_full_body`)

**Goal**: match the “same” mole across two full-body images and compare ABCD over time.

**Backend steps** (`compare_loftr_pipeline.py`):

1. **Detect moles in both images (YOLO)**
2. **Compute detection centers**
   - Convert YOLO boxes to center points (in pixels).
3. **Match across images using LoFTR**
   - LoFTR finds many correspondences between the two full-body images.
   - The code ties LoFTR correspondences to YOLO center points within a pixel radius,
     then infers which detection in image B corresponds to each detection in image A.
4. **Crop both matched moles**
   - Crop mole A and mole B using their YOLO boxes.
   - Upscale crops using Real-ESRGAN (to match full-body pipeline quality).
5. **Analyze each crop**
   - Run the integrated segmentation+ABCD pipeline on both crops.
6. **Compute % change per matched pair**
7. **Response**
   - Returns a list of matched pairs with:
     - `bbox_a`, `bbox_b`
     - `cropped_a`, `cropped_b`
     - `metrics_a`, `metrics_b`
     - `percent_change`

**Frontend behavior**:

- Renders pairs side-by-side (crop A vs crop B), then metrics and percent change.

---

## How ABCD is computed (conceptually)

The app computes ABCD-style features from:

- The **original RGB image** (for colour features)
- The **binary segmentation mask** (for geometry/shape features)

Representative logic is shown in `metrics/improved_metrics.py`:

- **Asymmetry**
  - Rotate mask 180° around centroid.
  - Compare overlap:
    - \( 1 - \frac{\text{intersection}}{\text{union}} \)
  - Scaled to a 0–10 style score in that module.

- **Border**
  - Uses perimeter \(P\) and area \(A\) of the lesion mask.
  - Circularity-based irregularity \( \frac{P^2}{4\pi A} \) (1 is circle; higher is more irregular), then scaled.

- **Diameter**
  - Uses a maximum caliper distance (Feret diameter) from `skimage.measure.regionprops`.

- **Colour**
  - Converts lesion pixels to HSV.
  - Uses standard deviation of Hue over lesion pixels as a proxy for colour variation, then scales.

Note: The integrated pipeline uses `metrics/merged_improved_metrics.py` (`MoleAnalyzer`) as its primary implementation.

---

## Tech stack / tools used and why

### Backend language and framework

- **Python**
  - Best ecosystem for CV/ML (PyTorch, OpenCV, skimage, numpy).
  - Keeps model inference and numeric feature extraction in one place.

- **Flask**
  - Lightweight web server for:
    - HTML pages (templates)
    - JSON APIs for image processing
  - Simple deployment and straightforward request handling for file uploads.

### Computer vision + numerical stack

- **OpenCV (`cv2`)**
  - Image read/write, resizing/cropping, overlays, geometry, performance.

- **NumPy**
  - Array operations, mask logic, numeric calculations.

- **scikit-image (`skimage`)**
  - `perimeter`, `regionprops` for shape features (border, Feret diameter).

- **SciPy**
  - Useful for image processing utilities (rotation, filtering) in some metrics implementations.

### Deep learning models

- **PyTorch**
  - Foundation for the ML models.

- **YOLOv5 (via `torch.hub`)**
  - Fast and accurate object detection, suited to **detecting multiple moles** in high-resolution full-body images.
  - Custom weights allow training specifically for mole detection.

- **MobileUNETR segmentation**
  - Produces a precise lesion mask needed for reliable ABCD measurements.
  - UNet-style segmentation is a standard approach in medical imaging.

- **Real-ESRGAN / DermaRealESRGAN x2**
  - Super-resolution improves the detail in small crops, which can help both:
    - perceived UI quality (sharper thumbnails),
    - and potentially segmentation/metric stability on low-detail crops.

- **LoFTR**
  - Robust feature matching between two scans without hand-crafted keypoints.
  - Used to pair corresponding moles across time in full-body comparisons (when body pose/lighting differs).

### Frontend/UI stack

- **Jinja templates (`templates/*.html`)**
  - Server-rendered pages; simple and easy to iterate.

- **Tailwind CSS**
  - Fast UI iteration with utility classes (layout, spacing, responsiveness).
  - Good fit when you want to rapidly prototype/iterate UI/UX.

- **Vanilla JavaScript**
  - Handles file uploads (`fetch` + `FormData`) and dynamic rendering of results.
  - Avoids introducing a heavier framework for a relatively bounded UI.

---

## Data flow summary (one-screen)

1. **User uploads image(s)** in the browser.
2. **Flask** receives upload → saves into `uploads/`.
3. A pipeline runs:
   - Single: segmentation → ABCD
   - Full-body: YOLO detect → crop → (upscale) → segmentation → ABCD
   - Comparisons: run above twice + compute % change
   - Full-body comparison: YOLO both → LoFTR pair → crop pairs → segmentation → ABCD → % change
4. Flask returns **JSON + URLs** to generated images in `outputs/` / `full_body_output/`.
5. Browser renders:
   - Images (original/mask/overlay/crops)
   - ABCD scores
   - comparisons (percent change)

---


