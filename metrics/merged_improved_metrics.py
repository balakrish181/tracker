import numpy as np
import imageio
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage
from skimage.measure import perimeter
from itertools import combinations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


class MoleAnalyzer:
    def __init__(self, original_img_path, binary_mask_path):
        self.original_img = imageio.imread(original_img_path)
        # Ensure original image is RGB
        if len(self.original_img.shape) == 2:  # Grayscale
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2BGR)
            
        self.mask = imageio.imread(binary_mask_path)
        self.mask = self.prepare_mask(self.mask)
        self.masked_img = self.mask_image(self.original_img, self.mask)
        self.result_img_path = "Result.jpg"
        imageio.imwrite(self.result_img_path, self.masked_img)

    @staticmethod
    def prepare_mask(mask):
        """Ensure the binary mask is in boolean format."""
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Use one channel if RGB
        return mask > 127  # Convert to boolean

    @staticmethod
    def mask_image(image, mask):
        result = np.zeros_like(image)
        if image.ndim == 3:
            # Ensure mask is broadcastable if image is RGB
            mask_3channel = np.stack([mask]*3, axis=-1) if image.ndim == 3 else mask
            result = np.where(mask_3channel, image, 0)
        else:
            result = np.where(mask, image, 0)
        return result

    def calculate_area(self, mask):
        """Calculate the number of pixels in the lesion and background."""
        flat = mask.flatten()
        affect = np.sum(flat != 0)
        naffect = flat.size - affect
        return naffect - affect, affect
        
    def compute_asymmetry(self, mask_uint8):
        """
        Calculates asymmetry based on 180-degree rotation around centroid.
        
        This improved method measures asymmetry by rotating the lesion mask 180 degrees
        around its centroid and calculating the mismatch between the original and rotated masks.
        The ratio of intersection to union gives a measure of similarity, which is inverted to get asymmetry.
        
        Formula: Asymmetry = 1 - (Area of Intersection / Area of Union)
        
        Score interpretation:
        - 0 = perfectly symmetric (original and rotated masks completely overlap)
        - 1 = completely asymmetric (no overlap between original and rotated masks)
        
        Parameters:
        ----------
        mask_uint8 : numpy.ndarray
            Binary mask of the lesion as an 8-bit image (values 0 and 255)
            
        Returns:
        -------
        float
            Asymmetry score in the range [0, 1]
        """
        if np.sum(mask_uint8) == 0:
            return 0.0

        # Find centroid
        M = cv2.moments(mask_uint8)
        if M["m00"] == 0:  # Should not happen if sum > 0, but as a safeguard
            return 0.0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Rotate mask 180 degrees around centroid
        rows, cols = mask_uint8.shape
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), 180, 1)
        rotated_mask = cv2.warpAffine(mask_uint8, rotation_matrix, (cols, rows))

        # Ensure rotated mask is binary (0 or 1) after interpolation
        rotated_mask_binary = (rotated_mask > 0.5).astype(np.uint8)

        intersection = np.sum(np.logical_and(mask_uint8, rotated_mask_binary))
        union = np.sum(np.logical_or(mask_uint8, rotated_mask_binary))

        if union == 0:
            return 0.0  # Or 1.0 if interpreted as completely non-overlapping (though unlikely)
        
        asymmetry_index = 1.0 - (intersection / union)
        return asymmetry_index

    def border_irregularity_index(self, mask_uint8):
        """
        Calculate the border irregularity index using the ratio of perimeter to circle perimeter
        with the same area.
        
        This improved method quantifies border irregularity by comparing the actual perimeter
        of the lesion contour to the perimeter of a perfect circle with the same area.
        The more irregular the border, the higher the ratio will be.
        
        Formula: Border Irregularity = Actual Perimeter / (2 * π * √(Area/π))
                 Where (2 * π * √(Area/π)) is the perimeter of a circle with the same area
        
        Score interpretation:
        - 1.0 = perfectly circular border (minimal irregularity)
        - >1.0 = increasing border irregularity (more jagged/irregular border)
        
        Parameters:
        ----------
        mask_uint8 : numpy.ndarray
            Binary mask of the lesion as an 8-bit image (values 0 and 255)
            
        Returns:
        -------
        float
            Border irregularity index (≥ 1.0, where 1.0 represents a perfect circle)
        """
        # Find contours of the lesion
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        # Get the largest contour (assumed to be the lesion)
        contour = max(contours, key=cv2.contourArea)

        # Calculate perimeter and area of the lesion
        perimeter_length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Calculate the radius of a circle with the same area
        radius = np.sqrt(area / np.pi)

        # Perimeter of a circle with the same area
        circle_perimeter = 2 * np.pi * radius

        # Border Irregularity Index: ratio of actual perimeter to circle perimeter
        irregularity_index = perimeter_length / circle_perimeter if circle_perimeter > 0 else 0

        return irregularity_index

    def calculate_diameter(self, mask_uint8):
        """
        Calculate the longest distance (Feret diameter) of the lesion in pixels
        
        This improved method determines the maximum diameter of the lesion by finding the
        longest distance between any two points on the lesion contour. This is known as
        the Feret diameter or maximum caliper distance.
        
        The method examines all pairs of points on the contour to find the maximum distance,
        which provides a more accurate measure of lesion size than simple width or height.
        
        Parameters:
        ----------
        mask_uint8 : numpy.ndarray
            Binary mask of the lesion as an 8-bit image (values 0 and 255)
            
        Returns:
        -------
        float
            Maximum diameter of the lesion in pixels
        """
        # Find contours of the lesion
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, None, None, None

        # Get the largest contour (assumed to be the lesion)
        contour = max(contours, key=cv2.contourArea)

        # Extract contour points
        contour_points = contour.reshape(-1, 2)  # Shape: (N, 2) where N is number of points

        # Calculate distances between all pairs of points
        max_distance = 0
        pt1, pt2 = None, None
        for p1, p2 in combinations(contour_points, 2):
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if distance > max_distance:
                max_distance = distance
                pt1, pt2 = p1, p2

        return max_distance

    def color_space_analysis(self, original_img, mask):
        """
        Analyze color variance in the lesion using L*a*b* color space
        
        This improved method analyzes color variation within the lesion by converting
        the image to the L*a*b* color space, which better represents how humans perceive
        color differences. The L* channel represents lightness, the a* channel represents
        green-red, and the b* channel represents blue-yellow.
        
        The method calculates the variance in each channel for lesion pixels only and 
        sums them to get a total color variance. This provides a more comprehensive
        measure of color heterogeneity than using just the hue channel from HSV.
        
        Parameters:
        ----------
        original_img : numpy.ndarray
            Original RGB image
        mask : numpy.ndarray
            Boolean mask where True represents lesion pixels
            
        Returns:
        -------
        float
            Total variance across L*a*b* channels, representing color heterogeneity
        """
        # Apply mask to isolate the lesion
        mask_3ch = np.stack([mask]*3, axis=-1) if original_img.ndim == 3 else mask
        lesion_img = cv2.bitwise_and(original_img, mask_3ch.astype(np.uint8) * 255)
        
        # Convert to LAB color space
        img_lab = cv2.cvtColor(lesion_img, cv2.COLOR_BGR2LAB)
        
        # Apply mask to L*a*b* image to focus on lesion pixels only
        mask_binary = mask.astype(np.uint8)  # Convert to 0 and 1
        mask_expanded = np.expand_dims(mask_binary, axis=2)  # Shape to (H, W, 1)
        mask_expanded = np.repeat(mask_expanded, 3, axis=2)  # Shape to (H, W, 3)
        img_lab_masked = img_lab * mask_expanded  # Zero out non-lesion pixels
        
        # Calculate variance for lesion pixels only
        lesion_pixels = img_lab_masked[np.where(mask_expanded == 1)]
        lesion_pixels = lesion_pixels.reshape(-1, 3)
        
        if len(lesion_pixels) == 0:
            return 0
        
        lab_variances = np.var(lesion_pixels, axis=0)
        total_variance = np.sum(lab_variances)
        
        return total_variance

    def analyze(self, show=True):
        img = self.masked_img
        mask_uint8 = self.mask.astype(np.uint8) * 255

        # Area calculation
        _, A = self.calculate_area(self.mask)
        
        # Asymmetry calculation
        asymmetry_raw = self.compute_asymmetry(mask_uint8)
        Asymmetry = asymmetry_raw * 10  # Scale to 0-10 range

        # Border irregularity calculation
        border_raw = self.border_irregularity_index(mask_uint8)
        Border = border_raw / 10  # Scale to match original scaling

        # Diameter calculation (Feret diameter)
        diameter_raw = self.calculate_diameter(mask_uint8)
        Diameter = diameter_raw / 10  # Scale to match original scaling

        # Color variance calculation
        if len(img.shape) == 3:
            hsv_img = matplotlib.colors.rgb_to_hsv(img / 255.0)
            Colour = self.calculate_colour_sd(hsv_img)
        else:
            # For grayscale images or if HSV conversion is not applicable
            color_variance = self.color_space_analysis(self.original_img, self.mask)
            Colour = color_variance / 100  # Scaling to match original method

        if show:
            print(f"Asymmetry: {Asymmetry:.2f}")
            print(f"Border: {Border:.2f}")
            print(f"Diameter: {Diameter:.2f}")
            print(f"Colour: {Colour:.2f}")
            plt.title("Masked Lesion")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
            plt.axis('off')
            plt.show()
            
        # Convert NumPy types to native Python types for JSON serialization
        A_py = int(A) if isinstance(A, np.integer) else float(A)
        asymmetry_raw_py = float(asymmetry_raw) if isinstance(asymmetry_raw, (np.float32, np.float64)) else asymmetry_raw
        Asymmetry_py = float(Asymmetry) if isinstance(Asymmetry, (np.float32, np.float64)) else Asymmetry
        Border_py = float(Border) if isinstance(Border, (np.float32, np.float64)) else Border
        border_raw_py = float(border_raw) if isinstance(border_raw, (np.float32, np.float64)) else border_raw
        Diameter_py = float(Diameter) if isinstance(Diameter, (np.float32, np.float64)) else Diameter
        diameter_raw_py = float(diameter_raw) if isinstance(diameter_raw, (np.float32, np.float64)) else diameter_raw
        Colour_py = float(Colour) if isinstance(Colour, (np.float32, np.float64)) else Colour

        return {
            "Asymmetry": Asymmetry_py,
            "Border": Border_py,
            "Diameter": Diameter_py,
            "Colour": Colour_py,
            "Raw_Metrics": {
                "Area_pixels": A_py,
                "Asymmetry_0_1": asymmetry_raw_py,
                "Border_CircularityIndex": border_raw_py,
                "Diameter_Feret_pixels": diameter_raw_py
            }
        }
        
    def calculate_colour_sd(self, hsv_img):
        """
        Original method for color analysis using HSV hue standard deviation
        
        This method calculates the standard deviation of the hue channel (from HSV color space)
        for lesion pixels only. The standard deviation measures the spread of colors around the
        mean hue value, with higher values indicating more color variation/heterogeneity.
        
        Note: While this method provides a simple measure of color variation, it only considers
        the hue channel and may miss important variations in saturation and value. The improved
        color_space_analysis method using L*a*b* space is often more comprehensive.
        
        Parameters:
        ----------
        hsv_img : numpy.ndarray
            Image in HSV color space
            
        Returns:
        -------
        float
            Standard deviation of hue channel values within the lesion, scaled by 1/10
        """
        hue = hsv_img[:, :, 0]
        # Take mean only of the masked pixels
        lesion_pixels = hue[self.mask]
        if len(lesion_pixels) == 0:
            return 0
        mean = np.mean(lesion_pixels)
        sd = np.sqrt(np.sum((lesion_pixels - mean) ** 2)) / (len(lesion_pixels) - 1) if len(lesion_pixels) > 1 else 0
        return sd / 10  # Match original scaling


if __name__ == "__main__":
    analyzer = MoleAnalyzer("test_images/image.png", "test_images/image_mask.png")
    results = analyzer.analyze()
    print(results)
