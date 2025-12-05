import os
from pathlib import Path
import cv2
import numpy as np

from full_body_pipeline import FullBodyMoleAnalysisPipeline
from integrated_pipeline import IntegratedMolePipeline
from loftr.loftr_matcher import LoFTRMatcher
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
            rimg1 = cv2.resize(img1, (int(w1 * s1), int(h1 * s1)), interpolation=cv2.INTER_AREA)
            t1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            t1.close()
            cv2.imwrite(t1.name, rimg1)
            use_path1 = t1.name
            tmp1 = t1.name
        if s2 != 1.0:
            rimg2 = cv2.resize(img2, (int(w2 * s2), int(h2 * s2)), interpolation=cv2.INTER_AREA)
            t2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            t2.close()
            cv2.imwrite(t2.name, rimg2)
            use_path2 = t2.name
            tmp2 = t2.name

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
        for i, (cx, cy) in enumerate(c1_scaled):
            if len(mk0) == 0:
                continue
            d = np.linalg.norm(mk0 - np.array([cx, cy]), axis=1)
            idxs = np.where(d <= scaled_radius)[0]
            if len(idxs) == 0:
                continue
            target_pts = mk1[idxs]
            if len(c2) == 0:
                continue
            c2arr = np.array(c2_scaled)
            d2 = np.linalg.norm(target_pts[:, None, :] - c2arr[None, :, :], axis=2)
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
            def g(d, k):
                v = d.get(k)
                return float(v) if isinstance(v, (int,float)) else None
            def pct(a, b):
                if a is None or b is None:
                    return None
                if a == 0:
                    return None
                return ((b - a) / a) * 100.0
            keys = ["Asymmetry","Border","Colour","Diameter"]
            pct_dict = {k: pct(g(m1,k), g(m2,k)) for k in keys}
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
