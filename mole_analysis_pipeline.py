import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from seg_mole_metrics.mobileunetr import build_mobileunetr_xxs
from seg_mole_metrics.inference import MobileUNETRInference
from metrics.merged_improved_metrics import MoleAnalyzer
from realesrgan_upscaler import DermaRealESRGANx2

class MoleAnalysisPipeline:
    def __init__(self, model_path="weights/segment_mob_unet_.bin"):
        """Initialize the pipeline with the segmentation model."""
        self.segmentation_model = MobileUNETRInference(model_path)
        
    def process_image(self, image_path, save_outputs=True, output_dir="outputs"):
        """
        Process a single mole image through the complete pipeline.
        
        Args:
            image_path (str): Path to the input image
            save_outputs (bool): Whether to save intermediate and final results
            output_dir (str): Directory to save outputs
            
        Returns:
            dict: Dictionary containing segmentation mask and ABCD metrics
        """
        # Create output directory if it doesn't exist
        if save_outputs:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Upscale image if needed
        upscaler = DermaRealESRGANx2(model_path='weights/dermaRealESRGAN_x2plus_v1.pth', fp32=True)
        upscale_img_path = f"{output_dir}/{Path(image_path).stem}_upscaled.png"
        upscaler.upscale(image_path, upscale_img_path)
        image_path = upscale_img_path
        
        # 1. Generate segmentation mask
        mask = self.segmentation_model.predict(image_path)
        
        # Save mask if requested
        if save_outputs:
            base_name = Path(image_path).stem
            mask_path = f"{output_dir}/{base_name}_mask.png"
            cv2.imwrite(mask_path, mask)
            
            # Create and save overlay
            original_img = cv2.imread(image_path)
            original_img = cv2.resize(original_img, (mask.shape[1], mask.shape[0]))
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(f"{output_dir}/{base_name}_overlay.png", overlay)
        
        # 2. Calculate ABCD metrics
        analyzer = MoleAnalyzer(image_path, mask_path if save_outputs else mask)
        metrics = analyzer.analyze(show=False)
        
        # Save results to file if requested
        if save_outputs:
            with open(f"{output_dir}/{base_name}_metrics.txt", "w") as f:
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        f.write(f"{metric}:\n")
                        for sub_metric, sub_value in value.items():
                            f.write(f"  {sub_metric}: {sub_value:.2f}\n")
                    else:
                        f.write(f"{metric}: {value:.2f}\n")
        
        return {
            "mask": mask,
            "metrics": metrics
        }

def main():
    # Example usage
    pipeline = MoleAnalysisPipeline()
    
    # Process a test image
    image_path = "test_images/image.png"
    results = pipeline.process_image(image_path)
    
    # Print results
    print("\nABCD Metrics:")
    for metric, value in results["metrics"].items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value:.2f}")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()