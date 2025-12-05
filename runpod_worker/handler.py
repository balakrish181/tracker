import os
from pathlib import Path

import runpod

from integrated_pipeline import IntegratedMolePipeline
from full_body_pipeline import FullBodyMoleAnalysisPipeline
from runpod_worker.utils_io import load_image_to_path, image_file_to_b64

# Resolve weights directory relative to this file or project root
ROOT = Path(__file__).resolve().parents[1]  # .../version1
WEIGHTS_DIR = ROOT / "weights"

SEG_WEIGHTS = str(WEIGHTS_DIR / "segment_mob_unet_.bin")
YOLO_WEIGHTS = str(WEIGHTS_DIR / "best_1280_default_hyper.pt")
# DermaRealESRGAN path is referenced inside full_body_pipeline as 'weights/dermaRealESRGAN_x2plus_v1.pth'

# Ensure YOLO uses local repo if present (set via Dockerfile)
YOLOV5_DIR = os.getenv("YOLOV5_DIR")

# Initialize pipelines once per worker
single_pipeline = IntegratedMolePipeline(model_path=SEG_WEIGHTS)
full_pipeline = FullBodyMoleAnalysisPipeline(
    yolo_model_path=YOLO_WEIGHTS,
    segmentation_model_path=SEG_WEIGHTS,
)


def handle_analyze(inp: dict):
    image_b64 = inp.get("image_b64")
    image_url = inp.get("image_url")
    return_images = bool(inp.get("return_images", False))

    img_path, (w, h) = load_image_to_path(image_b64=image_b64, image_url=image_url)

    base_name = Path(img_path).stem
    out_dir = "/tmp"
    results = single_pipeline.process_image(img_path, save_intermediate=True, output_dir=out_dir)

    response = {
        "metrics": results,
        "image_dimensions": {"width": w, "height": h},
    }

    if return_images:
        mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")
        artifacts = {}
        if os.path.exists(mask_path):
            artifacts["mask_b64"] = image_file_to_b64(mask_path)
        if os.path.exists(overlay_path):
            artifacts["overlay_b64"] = image_file_to_b64(overlay_path)
        response["artifacts"] = artifacts

    return response


def handle_analyze_full_body(inp: dict):
    image_b64 = inp.get("image_b64")
    image_url = inp.get("image_url")
    return_images = bool(inp.get("return_images", False))

    img_path, (w, h) = load_image_to_path(image_b64=image_b64, image_url=image_url)

    out_dir = "/tmp"
    results = full_pipeline.process_full_body_image(img_path, output_dir=out_dir)

    if return_images:
        for item in results:
            path = item.get("cropped_image_path")
            if path and os.path.exists(path):
                item["cropped_b64"] = image_file_to_b64(path)

    return {
        "results": results,
        "image_dimensions": {"width": w, "height": h},
    }


@runpod.serverless.functions.handler
def handler(event):
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    task = inp.get("task")

    if task == "analyze":
        return handle_analyze(inp)
    elif task == "analyze_full_body":
        return handle_analyze_full_body(inp)
    else:
        raise ValueError("Unsupported task. Use 'analyze' or 'analyze_full_body'.")


if __name__ == "__main__":
    # Start the RunPod serverless handler loop when running in container
    runpod.serverless.start()
