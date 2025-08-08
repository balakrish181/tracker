import numpy as np
import imageio
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage
from skimage.measure import perimeter

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


class MoleAnalyzer:
    def __init__(self, original_img_path, binary_mask_path):
        self.original_img = imageio.imread(original_img_path)
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
            for i in range(3):
                result[:, :, i] = np.where(mask, image[:, :, i], 0)
        else:
            result = np.where(mask, image, 0)
        return result

    def calculate_area(self, mask):
        flat = mask.flatten()
        affect = np.sum(flat != 0)
        naffect = flat.size - affect
        return naffect - affect, affect

    def calculate_colour_sd(self, hsv_img):
        hue = hsv_img[:, :, 0]
        mean = np.mean(hue) # Should we take mean only of the masked pixels or should we include the whole image?
        sd = np.sqrt(np.sum((hue - mean) ** 2)) / (hue.size - 1)
        return sd

    def analyze(self, show=True):
        img = self.masked_img
        mask = self.mask.astype(np.uint8)

        # A = number of lesion pixels
        # a = naffect - affect
        a, A = self.calculate_area(mask)
        Asymmetry = abs((a / A) * 100 / 10) if A != 0 else 0

        P = perimeter(mask, neighborhood=8)

        Border = ((P ** 2) / (4 * math.pi * A)) / 10 if A > 0 else 0  #circularity index


        n = (4 * A) / P if P != 0 else 0 
        Diameter = math.sqrt(n) / 10 if n > 0 else 0

        hsv_img = matplotlib.colors.rgb_to_hsv(img / 255.0)
        Colour = self.calculate_colour_sd(hsv_img) / 10

        if show:
            print(f"Asymmetry: {Asymmetry:.2f}")
            print(f"Border: {Border:.2f}")
            print(f"Diameter: {Diameter:.2f}")
            print(f"Colour: {Colour:.2f}")
            plt.title("Masked Lesion")
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        return {
            "Asymmetry": Asymmetry,
            "Border": Border,
            "Diameter": Diameter,
            "Colour": Colour
        }


if __name__ == "__main__":
    analyzer = MoleAnalyzer("test_images/image.png", "test_images/image_mask.png")
    results = analyzer.analyze()
    print(results)
