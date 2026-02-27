# Mole Analysis Software — Non‑Technical Overview

---

## What this software is

This is an image-based tool that helps **screen and track skin moles** by:

- Identifying moles in photos
- Measuring visual characteristics using the **ABCD** method
- Presenting results as easy-to-review metrics and visuals
- Comparing changes over time (single mole or full-body scans)

It is designed to support **triage, monitoring, and structured reporting** from images.

---

## Who it’s for

- **Clinicians / reviewers**: to quickly review mole characteristics and changes.
- **Operators / technicians**: to upload images and generate reports.
- **Product / operations stakeholders**: to understand workflows and outputs.

---

## Main functions (what users can do)

### 1) Single Mole Analysis

**What the user does**

- Uploads **one** close-up image of a mole.

**What the software returns**

- A **segmentation mask** (highlights the lesion area)
- An **overlay image** (visual check of the segmented region)
- **ABCD metrics**:
  - Asymmetry
  - Border
  - Colour
  - Diameter

**Typical use**

- Quick assessment of a single lesion from a focused photo.

---

### 2) Full‑Body Screening (multi-mole detection)

**What the user does**

- Uploads **one** full-body photo.

**What the software returns**

- The original image with **bounding boxes** (mole locations)
- A list/grid of detected moles
- For each detected mole:
  - A cropped mole image
  - ABCD metrics

**Typical use**

- One scan → many moles → fast review and prioritization.

---

### 3) Single‑Mole Comparison (two images)

**What the user does**

- Uploads **two** images of a mole (e.g., earlier vs later).

**What the software returns**

- ABCD metrics for each image
- **Percent change** for each metric (how much it increased/decreased)
- Side-by-side presentation for quick review

**Typical use**

- Tracking one lesion over time.

---

### 4) Full‑Body Longitudinal Comparison (two full-body scans)

**What the user does**

- Uploads **two** full-body photos from different times.

**What the software returns**

- Automatically matched “pairs” of likely corresponding moles across the two scans
- For each matched pair:
  - Cropped image from scan 1 vs cropped image from scan 2
  - ABCD metrics for each
  - Percent change in ABCD metrics

**Typical use**

- Monitoring changes across many lesions between visits.

---

## What “ABCD” means (simple explanation)

ABCD is a common way to describe suspicious visual characteristics of skin lesions:

- **A — Asymmetry**: how uneven the shape appears.
- **B — Border**: how irregular or uneven the edge appears.
- **C — Colour**: how varied the colours are within the lesion.
- **D — Diameter**: how large the lesion appears in the image.

The software converts these into **numeric scores** to help compare and track patterns consistently.

---

## What users see as outputs

Depending on the workflow, the results may include:

- **Visuals**
  - Original image preview
  - Segmentation mask and overlay (single mole)
  - Bounding boxes on full-body images
  - Cropped mole thumbnails/cards
- **Numbers**
  - ABCD metric values
  - Percent change between two timepoints (comparison modes)

These outputs are intended to make review faster and more standardized.

---

## Important notes (non-technical)

- **Image quality matters**: lighting, focus, and framing affect results.
- **Results support review**: the tool is designed to aid analysis and tracking, not replace clinical judgement.
- **Consistency helps longitudinal tracking**: using similar camera distance/angle improves comparisons over time.

---


