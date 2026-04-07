import os
from pathlib import Path

from integrated_pipeline import IntegratedMolePipeline
from constants import ABCD_KEYS, safe_get_metric, percent_change

class CompareMolePipeline:
    def __init__(self, integrated_pipeline: IntegratedMolePipeline | None = None, model_path: str | None = None):
        if integrated_pipeline is not None:
            self.pipeline = integrated_pipeline
        else:
            self.pipeline = IntegratedMolePipeline(model_path=model_path)

    def compare(self, image_path_1: str, image_path_2: str, output_dir: str | None = None):
        m1 = self.pipeline.process_image(image_path_1, save_intermediate=True, output_dir=output_dir)
        m2 = self.pipeline.process_image(image_path_2, save_intermediate=True, output_dir=output_dir)
        pct = {k: percent_change(safe_get_metric(m1, k), safe_get_metric(m2, k)) for k in ABCD_KEYS}
        return {
            "image1_metrics": m1,
            "image2_metrics": m2,
            "percent_change": pct,
        }
