import os
from pathlib import Path

from integrated_pipeline import IntegratedMolePipeline

class CompareMolePipeline:
    def __init__(self, integrated_pipeline: IntegratedMolePipeline | None = None, model_path: str | None = None):
        if integrated_pipeline is not None:
            self.pipeline = integrated_pipeline
        else:
            self.pipeline = IntegratedMolePipeline(model_path=model_path)

    def _safe_get(self, d, k):
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    def _pct_change(self, v1, v2):
        if v1 is None or v2 is None:
            return None
        if v1 == 0:
            return None
        return ((v2 - v1) / v1) * 100.0

    def compare(self, image_path_1: str, image_path_2: str, output_dir: str | None = None):
        m1 = self.pipeline.process_image(image_path_1, save_intermediate=True, output_dir=output_dir)
        m2 = self.pipeline.process_image(image_path_2, save_intermediate=True, output_dir=output_dir)
        keys = ["Asymmetry", "Border", "Colour", "Diameter"]
        pct = {k: self._pct_change(self._safe_get(m1, k), self._safe_get(m2, k)) for k in keys}
        return {
            "image1_metrics": m1,
            "image2_metrics": m2,
            "percent_change": pct,
            }
            

