import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .mobileunetr import build_mobileunetr_xxs
import cv2
import warnings; warnings.filterwarnings("ignore")

import pathlib

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

class MobileUNETRInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.config = {
            "model_name": "mobileunetr_xxs",
            "model_parameters": {
                "encoder": None,
                "bottle_neck": {
                    "dims": [96],
                    "depths": [3],
                    "expansion": 4,
                    "kernel_size": 3,
                    "patch_size": [2,2],
                    "channels": [80, 96, 96]
                },
                "decoder": {
                    "dims": [64, 80, 96],
                    "channels": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 96, 96, 320],
                    "num_classes": 1
                },
                "image_size": 512
            }
        }
        
        # Initialize model
        self.model = build_mobileunetr_xxs(config=self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess the input image."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            self.original_size = image.size[::-1]  # Store original size (height, width)
        else:
            image = Image.fromarray(image_path).convert('RGB')
            self.original_size = image_path.shape[:2]  # Store original size (height, width)
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def postprocess_mask(self, pred_mask):
        """Convert model output to binary mask."""
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Resize mask to original image size
        binary_mask = cv2.resize(binary_mask, (self.original_size[1], self.original_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
        return binary_mask
    
    def predict(self, image_path):
        """Generate segmentation mask for input image."""
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Get prediction
            pred_mask = self.model(image_tensor)
            
            # Postprocess mask
            binary_mask = self.postprocess_mask(pred_mask)
            
            return binary_mask

def main():
    # Example usage
    model_path = "weights\segment_mob_unet_.bin"
    inference = MobileUNETRInference(model_path)
    
    # Example with an image path
    image_path = "test_images/image.png"
    mask = inference.predict(image_path)
    
    # Save the binary mask
    import pathlib
    base_name = pathlib.Path(image_path).stem
    cv2.imwrite(f"test_images/{base_name}_mask.png", mask)

    # Load original image using OpenCV for overlay
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (mask.shape[1], mask.shape[0]))

    # Convert grayscale mask to BGR for overlay
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Blend original image and mask
    overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)

    # Save the overlay
    cv2.imwrite(f"test_images/{base_name}_overlay.png", overlay)
if __name__ == "__main__":
    main() 