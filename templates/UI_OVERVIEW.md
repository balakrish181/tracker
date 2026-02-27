# UI Overview (Design + Content Guide)

This document is for **non-technical stakeholders** and **designers**. It points to the exact UI files to edit for layout, copy, and styling.

---

## Where to make UI changes

All primary UI screens are HTML templates in:

- `version2/templates/`

Each page is mostly **Tailwind CSS classes** + **small built-in page scripts** that update the results area after you upload images.

---

## Screens (what users see) → files to edit

### Global layout (header / nav / page shell)

- **File**: `version2/templates/base.html`
- **Edit here for**:
  - App title / branding
  - Header layout and navigation labels
  - Shared UI elements (loading overlay, toast messages)

---

### Dashboard (entry page with tiles)

- **File**: `version2/templates/dashboard.html`
- **Edit here for**:
  - The main “tile” layout and copy (the cards that link to each workflow)
  - The order and wording of the feature entry points

---

### Single Mole Analysis (upload 1 image, show metrics + outputs)

- **File**: `version2/templates/single.html`
- **Key page sections**:
  - Upload form: `#single-form`
  - Results container: `#single-result`
- **Edit here for**:
  - Upload panel layout and instructional text
  - Results layout (image grid + metric cards)

---

### Full-Body Screening (upload full-body photo, see boxes + details)

- **File**: `version2/templates/full_body.html`
- **Key page sections**:
  - Upload form: `#fb-form`
  - Results view wrapper: `#fb-view`
  - Bounding box overlay layer: `#fb-boxes`
  - Details panel: `#fb-details`
- **Edit here for**:
  - The viewer layout (image area, controls, spacing)
  - The “mole details” card layout (metrics, risk badge, crop link)

---

### Single-Mole Comparison (upload 2 images, see side-by-side + percent change)

- **File**: `version2/templates/compare.html`
- **Key page sections**:
  - Upload form: `#cmp-form`
  - Results container: `#cmp-result`
- **Edit here for**:
  - Two-column upload layout
  - Comparison results layout (two images + percent change + metric panels)

---

### Full-Body Longitudinal Comparison (upload 2 full-body images, see matched pairs)

- **File**: `version2/templates/compare_full_body.html`
- **Key page sections**:
  - Upload form: `#fbc-form`
  - Results container: `#fbc-result`
- **Edit here for**:
  - Matched-pair card design (pair images, metrics columns, percent change block)
  - Spacing, typography, and information hierarchy for readability

---

## What’s safe to change (designer-friendly)

- **Copy**: headings, helper text, button labels, empty states.
- **Layout**: spacing, card structure, grids, alignment, responsive breakpoints.
- **Visual style**: Tailwind utility classes, colors, fonts, shadows, borders, badges.
- **Information hierarchy**: reorder blocks inside a page as long as the “anchor” containers still exist.

---

## What to avoid changing (to prevent breaking interactions)

The pages include small scripts that rely on specific elements being present.

- **Do not rename/remove these IDs** (examples):
  - `single.html`: `single-form`, `single-file`, `single-result`
  - `full_body.html`: `fb-form`, `fb-file`, `fb-view`, `fb-img`, `fb-boxes`, `fb-details`, `toggle-boxes`, `toggle-labels`
  - `compare.html`: `cmp-form`, `cmp-file1`, `cmp-file2`, `cmp-result`
  - `compare_full_body.html`: `fbc-form`, `fbc-file1`, `fbc-file2`, `fbc-result`
- **Avoid removing the results containers** (the scripts render results into them).
- **If you must rename/restructure IDs**, coordinate with engineering so the script hooks are updated too.

---

## Quick preview workflow (for stakeholders)

- Run the app locally, open the Dashboard, and navigate to the screen you’re reviewing.
- Make changes in the relevant file under `version2/templates/`.
- Refresh the browser page to see updated layout/copy immediately.

