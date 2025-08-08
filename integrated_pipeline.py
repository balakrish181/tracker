import os
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import segmentation model
from seg_mole_metrics.inference import MobileUNETRInference
# Import mole analysis
from metrics.merged_improved_metrics import MoleAnalyzer

class IntegratedMolePipeline:
    """
    Integrated pipeline that combines segmentation and ABCD analysis into a single workflow.
    This class integrates the MobileUNETRInference model for segmentation and 
    the MoleAnalyzer for ABCD scoring.
    """
    def __init__(self, model_path=None):
        """
        Initialize the integrated pipeline.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the segmentation model weights file.
            If None, will look for the weights file in the default location.
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "weights", "segment_mob_unet_.bin")
        
        # Ensure model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at: {model_path}")
            
        # Initialize segmentation model
        self.segmentation_model = MobileUNETRInference(model_path)
        
    def process_image(self, image_path, save_intermediate=True, output_dir=None):
        """
        Process an image through the complete pipeline: segmentation followed by ABCD analysis.
        
        Parameters:
        -----------
        image_path : str
            Path to the input image file.
        save_intermediate : bool, optional
            Whether to save intermediate results (mask, masked image).
        output_dir : str, optional
            Directory to save intermediate results. If None, uses the same directory as the input image.
            
        Returns:
        --------
        dict
            ABCD analysis results including asymmetry, border, color, and diameter scores.
        """
        # Validate input image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at: {image_path}")
            
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
            
        # Step 1: Generate binary mask with the segmentation model
        binary_mask = self.segmentation_model.predict(image_path)
        
        # Get base filename without extension
        base_name = Path(image_path).stem
        
        # Save binary mask if requested
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        if save_intermediate:
            cv2.imwrite(mask_path, binary_mask)
        
        # Step 2: Perform ABCD analysis using the original image and binary mask
        # Since MoleAnalyzer expects file paths, we need to save the mask temporarily if not already saved
        if not save_intermediate:
            cv2.imwrite(mask_path, binary_mask)
            
        # Initialize and run the analyzer
        try:
            analyzer = MoleAnalyzer(image_path, mask_path)
            results = analyzer.analyze(show=False)
            
            # Create an overlay visualization if saving intermediates
            if save_intermediate:
                # Load original image
                original_img = cv2.imread(image_path)
                
                # Resize if necessary to match mask dimensions
                if original_img.shape[:2] != binary_mask.shape[:2]:
                    original_img = cv2.resize(original_img, (binary_mask.shape[1], binary_mask.shape[0]))
                
                # Create colored mask overlay
                mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
                
                # Save the overlay
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay)
                
            # Clean up temporary mask file if not saving intermediates
            if not save_intermediate and os.path.exists(mask_path):
                os.remove(mask_path)
                
            return results
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            # Clean up temporary mask file if it exists
            if not save_intermediate and os.path.exists(mask_path):
                os.remove(mask_path)
            raise
    
    def analyze_image_batch(self, image_paths, save_intermediate=False, output_dir=None):
        """
        Process multiple images through the pipeline.
        
        Parameters:
        -----------
        image_paths : list of str
            List of paths to input images.
        save_intermediate : bool, optional
            Whether to save intermediate results.
        output_dir : str, optional
            Directory to save results. If None, uses the same directories as input images.
            
        Returns:
        --------
        dict
            Dictionary mapping image paths to their analysis results.
        """
        results = {}
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, save_intermediate, output_dir)
                results[image_path] = result
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results[image_path] = {"error": str(e)}
        return results


def main():
    """Example usage of the integrated pipeline."""
    # Initialize the pipeline
    pipeline = IntegratedMolePipeline()
    
    # Process a single image
    image_path = "test_images/image.png"
    try:
        results = pipeline.process_image(image_path, save_intermediate=True)
        print(f"Results for {image_path}:")
        print(f"  Asymmetry: {results['Asymmetry']:.2f}")
        print(f"  Border: {results['Border']:.2f}")
        print(f"  Colour: {results['Colour']:.2f}")
        print(f"  Diameter: {results['Diameter']:.2f}")
        print("Raw metrics:")
        for key, value in results["Raw_Metrics"].items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
