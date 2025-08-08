import numpy as np
import imageio
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Not strictly used in the final code, but often useful
import matplotlib
from scipy import ndimage # For rotation
from skimage.measure import perimeter, regionprops

from pathlib import Path
import sys

# Assuming the script is in a subdirectory, and the parent directory should be in sys.path
# Adjust if your project structure is different
# sys.path.append(str(Path(__file__).resolve().parent.parent))



class MoleAnalyzerImproved:
    def __init__(self, original_img_path, binary_mask_path):
        self.original_img_rgb = imageio.v2.imread(original_img_path) # Use imageio.v2
        # Ensure original image is RGB
        if self.original_img_rgb.ndim == 2: # Grayscale
            self.original_img_rgb = cv2.cvtColor(self.original_img_rgb, cv2.COLOR_GRAY2RGB)
        elif self.original_img_rgb.shape[2] == 4: # RGBA
            self.original_img_rgb = cv2.cvtColor(self.original_img_rgb, cv2.COLOR_RGBA2RGB)

        mask_raw = imageio.v2.imread(binary_mask_path) # Use imageio.v2
        self.boolean_mask = self.prepare_mask(mask_raw) # Keep as boolean for precise operations
        self.masked_img_display = self.mask_image_for_display(self.original_img_rgb, self.boolean_mask)

        self.result_img_path = "Result_Improved.jpg"
        imageio.v2.imwrite(self.result_img_path, self.masked_img_display) # Use imageio.v2

    @staticmethod
    def prepare_mask(mask_raw):
        """Ensure the binary mask is in boolean format."""
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]  # Use one channel if RGB
        return mask_raw > 127  # Convert to boolean (True for lesion, False for background)

    @staticmethod
    def mask_image_for_display(image_rgb, boolean_mask):
        """Creates an image with lesion on black background for display."""
        result = np.zeros_like(image_rgb)
        # Ensure mask is broadcastable if image is RGB
        mask_3channel = np.stack([boolean_mask]*3, axis=-1) if image_rgb.ndim == 3 else boolean_mask
        result = np.where(mask_3channel, image_rgb, 0)
        return result

    def calculate_lesion_area(self, boolean_mask):
        """Calculates the number of pixels in the lesion."""
        return np.sum(boolean_mask)

    def calculate_asymmetry_score(self, boolean_mask_uint8):
        """
        Calculates asymmetry based on 180-degree rotation around centroid.
        Score = 1 - (Area of Intersection / Area of Union)
        0 = perfectly symmetric, 1 = completely asymmetric.
        """
        if np.sum(boolean_mask_uint8) == 0:
            return 0.0

        # Find centroid
        M = cv2.moments(boolean_mask_uint8)
        if M["m00"] == 0: # Should not happen if sum > 0, but as a safeguard
            return 0.0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Rotate mask 180 degrees around centroid
        rows, cols = boolean_mask_uint8.shape
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), 180, 1)
        rotated_mask = cv2.warpAffine(boolean_mask_uint8, rotation_matrix, (cols, rows))

        # Ensure rotated mask is binary (0 or 1) after interpolation
        rotated_mask_binary = (rotated_mask > 0.5).astype(np.uint8)

        intersection = np.sum(np.logical_and(boolean_mask_uint8, rotated_mask_binary))
        union = np.sum(np.logical_or(boolean_mask_uint8, rotated_mask_binary))

        if union == 0:
            return 0.0  # Or 1.0 if interpreted as completely non-overlapping (though unlikely)
        
        asymmetry_index = 1.0 - (intersection / union)
        return asymmetry_index


    def calculate_diameter_score(self, boolean_mask_uint8):
        """Calculates the Feret diameter (maximum caliper diameter) in pixels."""
        if np.sum(boolean_mask_uint8) == 0:
            return 0.0
        
        # regionprops needs a labeled image. If mask is binary 0/1, label it.
        # If mask is already 0/lesion_label, it's fine.
        # For a single lesion, we can just pass the uint8 mask directly if lesion pixels are > 0.
        props = regionprops(boolean_mask_uint8) # Pass the uint8 mask
        if not props:
            return 0.0
        
        # Assuming a single connected component for the lesion, props[0] will be it.
        # If multiple components, this would need refinement (e.g., take largest).
        feret_diameter = props[0].feret_diameter_max
        return feret_diameter

    def calculate_colour_sds(self, original_rgb_image, boolean_mask):
        """
        Calculates Standard Deviation of H, S, V channels for LESION PIXELS ONLY.
        Returns a dictionary {'h_sd': sd_h, 's_sd': sd_s, 'v_sd': sd_v}
        """
        if np.sum(boolean_mask) == 0: # No lesion pixels
            return {'h_sd': 0.0, 's_sd': 0.0, 'v_sd': 0.0}

        hsv_img = matplotlib.colors.rgb_to_hsv(original_rgb_image / 255.0)
        
        lesion_pixels_h = hsv_img[boolean_mask, 0]
        lesion_pixels_s = hsv_img[boolean_mask, 1]
        lesion_pixels_v = hsv_img[boolean_mask, 2]

        if lesion_pixels_h.size <= 1: # Need more than 1 pixel to calculate std dev
             return {'h_sd': 0.0, 's_sd': 0.0, 'v_sd': 0.0}

        sd_h = np.std(lesion_pixels_h)
        sd_s = np.std(lesion_pixels_s)
        sd_v = np.std(lesion_pixels_v)
        
        return {'h_sd': sd_h, 's_sd': sd_s, 'v_sd': sd_v}


    def analyze(self, show=True):
        # Use the boolean mask for area and color, uint8 for perimeter, asymmetry, diameter
        mask_uint8 = self.boolean_mask.astype(np.uint8) # Convert boolean to 0/1

        # A = number of lesion pixels
        A = self.calculate_lesion_area(self.boolean_mask)
        
        # Asymmetry (0 to 1, 0=symmetric, 1=asymmetric)
        # The new asymmetry score is already normalized (0-1).
        # To scale it to a 0-10 range like the original, multiply by 10.
        Asymmetry_raw = self.calculate_asymmetry_score(mask_uint8)
        Asymmetry = Asymmetry_raw * 10 # Scale to 0-10 range

        # Perimeter
        P = perimeter(mask_uint8, neighborhood=8) # skimage.measure.perimeter

        # Border Irregularity (Circularity Index: 1 for circle, >1 for irregular)
        # Original formula: ((P ** 2) / (4 * math.pi * A))
        # This value is >= 1. To make it more like "irregularity", sometimes (Value - 1) is used.
        # Let's keep the original scaling for comparison.
        Border_raw = ((P ** 2) / (4 * math.pi * A)) if A > 0 else 0
        Border = Border_raw / 10 # Original scaling

        # Diameter (Feret diameter in pixels)
        # The raw Feret diameter can be large. The original scaling was /10.
        # This scaling might need adjustment based on typical image sizes and pixel resolutions
        # if a specific numeric range is desired.
        Diameter_raw = self.calculate_diameter_score(mask_uint8)
        Diameter = Diameter_raw / 10 # Original scaling, adjust as needed

        # Colour (Standard deviation of Hue for lesion pixels)
        # The SD of Hue (0-1 range) will be small.
        # Original scaling was /10. Let's apply a multiplier to make it more significant.
        # e.g., if Hue SD is 0.1, *100/10 = 1.0.
        colour_sds = self.calculate_colour_sds(self.original_img_rgb, self.boolean_mask)
        Colour_hue_sd_raw = colour_sds['h_sd']
        # Scale to be somewhat comparable to other 0-10 metrics.
        # Max Hue SD is around 0.5 (for uniform distribution). Let's scale by 20 to get 0-10.
        # Or, to match the old scaling logic: Colour_hue_sd_raw * 100 / 10
        Colour = Colour_hue_sd_raw * 100 / 10 # Scaling similar to original intent

        if show:
            print(f"Area (pixels): {A}")
            print(f"Perimeter (pixels): {P:.2f}")
            print(f"Asymmetry (0-10, higher is more asymmetric): {Asymmetry:.2f} (raw: {Asymmetry_raw:.3f})")
            print(f"Border (Circularity based, scaled): {Border:.2f} (raw: {Border_raw:.3f})")
            print(f"Diameter (Feret pixels, scaled): {Diameter:.2f} (raw: {Diameter_raw:.2f} pixels)")
            print(f"Colour (Hue SD within lesion, scaled): {Colour:.2f} (raw Hue SD: {Colour_hue_sd_raw:.3f})")
            print(f"  (Raw Saturation SD: {colour_sds['s_sd']:.3f}, Raw Value SD: {colour_sds['v_sd']:.3f})")
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(self.original_img_rgb)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Masked Lesion (for display)")
            plt.imshow(self.masked_img_display)
            plt.axis('off')
            plt.show()

        return {
            "Asymmetry": Asymmetry,
            "Border": Border,
            "Diameter": Diameter,
            "Colour_Hue_SD_Scaled": Colour, # Renamed for clarity
            "Raw_Metrics": { # Adding raw values for better understanding
                "Area_pixels": A,
                "Perimeter_pixels": P,
                "Asymmetry_0_1": Asymmetry_raw,
                "Border_CircularityIndex": Border_raw,
                "Diameter_Feret_pixels": Diameter_raw,
                "Colour_Hue_SD_lesion": Colour_hue_sd_raw,
                "Colour_Saturation_SD_lesion": colour_sds['s_sd'],
                "Colour_Value_SD_lesion": colour_sds['v_sd']
            }
        }

if __name__ == "__main__":
    

    analyzer_square = MoleAnalyzerImproved("test_images/image.png", "test_images/image_mask.png")
    results_square = analyzer_square.analyze()
    # print(results_square)
    print("\n")