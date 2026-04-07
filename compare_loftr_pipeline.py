import os
import logging
from pathlib import Path
import cv2
import numpy as np

from full_body_pipeline import FullBodyMoleAnalysisPipeline
from integrated_pipeline import IntegratedMolePipeline
from loftr.loftr_matcher import LoFTRMatcher
from constants import ABCD_KEYS, safe_get_metric, percent_change
import tempfile
from realesrgan_upscaler import DermaRealESRGANx2

class LoFTRFullBodyComparator:
    def __init__(self, yolo_model_path: str, seg_model_path: str, esrgan_model_path: str = 'weights/dermaRealESRGAN_x2plus_v1.pth', loftr_max_side: int = 960):
        self.fb = FullBodyMoleAnalysisPipeline(yolo_model_path=yolo_model_path, segmentation_model_path=seg_model_path)
        self.integrated = IntegratedMolePipeline(model_path=seg_model_path)
        self.matcher = LoFTRMatcher()
        self.loftr_max_side = loftr_max_side
        # Match full_body_pipeline behavior: DermaRealESRGAN x2 upscaling for crops
        self.upscaler = DermaRealESRGANx2(model_path=esrgan_model_path, fp32=True)

    def _bbox_centers_px(self, detections, w, h):
        centers = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = int(((x1 + x2) / 2.0) * w)
            cy = int(((y1 + y2) / 2.0) * h)
            centers.append((cx, cy))
        return centers

    def _pair_detections(self, img1_path, img2_path, dets1, dets2, radius_px=40):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        c1 = self._bbox_centers_px(dets1, w1, h1)
        c2 = self._bbox_centers_px(dets2, w2, h2)

        # Determine downscaling for LoFTR inputs
        s1 = 1.0
        s2 = 1.0
        max_side = getattr(self, 'loftr_max_side', None)
        if max_side:
            if max(h1, w1) > max_side:
                s1 = max_side / float(max(h1, w1))
            if max(h2, w2) > max_side:
                s2 = max_side / float(max(h2, w2))

        c1_scaled = [(int(cx * s1), int(cy * s1)) for (cx, cy) in c1]
        c2_scaled = [(int(cx * s2), int(cy * s2)) for (cx, cy) in c2]

        use_path1 = img1_path
        use_path2 = img2_path
        tmp1 = None
        tmp2 = None
        if s1 != 1.0:
            # rimg1 = cv2.resize(img1, (int(w1 * s1), int(h1 * s1)), interpolation=cv2.INTER_AREA)
            # t1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            # t1.close()

            #LOFTR expects size to be multiples of 8
            rimg1 = cv2.resize(img1, (int(w1 * s1), int(h1 * s1)), interpolation=cv2.INTER_AREA)
            h1r, w1r = rimg1.shape[:2]
            pad_h1 = (-h1r) % 8
            pad_w1 = (-w1r) % 8
            if pad_h1 or pad_w1:
                rimg1 = cv2.copyMakeBorder(rimg1, 0, pad_h1, 0, pad_w1, cv2.BORDER_CONSTANT, value=(0,0,0))
            logging.debug(f"LoFTR input1 shape: {rimg1.shape}")
            t1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            t1.close()
            cv2.imwrite(t1.name, rimg1)
            use_path1 = t1.name
            tmp1 = t1.name



        if s2 != 1.0:
            # rimg2 = cv2.resize(img2, (int(w2 * s2), int(h2 * s2)), interpolation=cv2.INTER_AREA)
            # t2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            # t2.close()

            rimg2 = cv2.resize(img2, (int(w2 * s2), int(h2 * s2)), interpolation=cv2.INTER_AREA)
            h2r, w2r = rimg2.shape[:2]
            pad_h2 = (-h2r) % 8
            pad_w2 = (-w2r) % 8
            if pad_h2 or pad_w2:
                rimg2 = cv2.copyMakeBorder(rimg2, 0, pad_h2, 0, pad_w2, cv2.BORDER_CONSTANT, value=(0,0,0))
            logging.debug(f"LoFTR input2 shape: {rimg2.shape}")
            t2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            t2.close()
            cv2.imwrite(t2.name, rimg2)
            use_path2 = t2.name
            tmp2 = t2.name

        # Warn if the two images have very different scales — LoFTR pixel matching degrades
        scale_ratio = max(s1, s2) / min(s1, s2) if min(s1, s2) > 0 else 1.0
        if scale_ratio > 1.5:
            logging.warning(
                f"Images have very different effective scales (ratio {scale_ratio:.2f}). "
                "LoFTR mole pairing may be unreliable. Try images of similar resolution."
            )

        scaled_radius = max(5, int(round(radius_px * s1)))

        m = self.matcher.match(use_path1, use_path2, yolov5_centers=c1_scaled, radius=scaled_radius)
        try:
            if 'filtered_matches_image0' in m:
                mk0 = np.array(m['filtered_matches_image0'])
                mk1 = np.array(m['filtered_matches_image1'])
            else:
                mk0 = np.array(m['matches_image0'])
                mk1 = np.array(m['matches_image1'])
        finally:
            if tmp1:
                try:
                    os.remove(tmp1)
                except Exception:
                    pass
            if tmp2:
                try:
                    os.remove(tmp2)
                except Exception:
                    pass
        pairs = {}
        if len(mk0) == 0 or len(mk1) == 0 or len(c2_scaled) == 0:
            logging.warning("LoFTR returned no matches or no detections in image2 — cannot pair moles")
            return []
        c2arr = np.array(c2_scaled)
        for i, (cx, cy) in enumerate(c1_scaled):
            d = np.linalg.norm(mk0 - np.array([cx, cy]), axis=1)
            idxs = np.where(d <= scaled_radius)[0]
            if len(idxs) == 0:
                continue
            target_pts = mk1[idxs]
            d2 = np.linalg.norm(target_pts[:, None, :] - c2arr[None, :, :], axis=2)
            if d2.size == 0:
                continue
            nearest = np.argmin(d2, axis=1)
            vals, counts = np.unique(nearest, return_counts=True)
            j = int(vals[np.argmax(counts)])
            pairs[i] = j
        used_b = set()
        final_pairs = []
        for a_idx, b_idx in pairs.items():
            if b_idx in used_b:
                continue
            used_b.add(b_idx)
            final_pairs.append((a_idx, b_idx))
        return final_pairs

    def _crop_single(self, image_path, det, output_dir):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        x1, y1, x2, y2, conf, cls = det
        ax1 = max(0, int(x1 * w))
        ay1 = max(0, int(y1 * h))
        ax2 = min(w, int(x2 * w))
        ay2 = min(h, int(y2 * h))
        crop = img[ay1:ay2, ax1:ax2]
        ch, cw = crop.shape[:2]
        
        if ch == 0 or cw == 0:
            return None

        base = Path(image_path).stem
        out_name = f"{base}_{ax1}_{ay1}_{ax2}_{ay2}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, crop)
        # Upscale crop directly (no padding)
        try:
            self.upscaler.upscale(out_path, out_path)
        except Exception:
            pass
        return out_path

    def compare(self, image1_path: str, image2_path: str, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dets1 = self.fb.detect_moles(image1_path)
        dets2 = self.fb.detect_moles(image2_path)
        pairs = self._pair_detections(image1_path, image2_path, dets1, dets2)
        results = []
        for a_idx, b_idx in pairs:
            d1 = dets1[a_idx]
            d2 = dets2[b_idx]
            c1_path = self._crop_single(image1_path, d1, output_dir)
            c2_path = self._crop_single(image2_path, d2, output_dir)
            if not c1_path or not c2_path:
                continue
            m1 = self.integrated.process_image(c1_path, save_intermediate=True, output_dir=output_dir)
            m2 = self.integrated.process_image(c2_path, save_intermediate=True, output_dir=output_dir)
            pct_dict = {k: percent_change(safe_get_metric(m1, k), safe_get_metric(m2, k)) for k in ABCD_KEYS}
            results.append({
                "a_index": int(a_idx),
                "b_index": int(b_idx),
                "bbox_a": [float(d1[0]), float(d1[1]), float(d1[2]), float(d1[3])],
                "bbox_b": [float(d2[0]), float(d2[1]), float(d2[2]), float(d2[3])],
                "cropped_a": c1_path,
                "cropped_b": c2_path,
                "metrics_a": m1,
                "metrics_b": m2,
                "percent_change": pct_dict
            })
        return {
            "pairs": results
        }
        














        