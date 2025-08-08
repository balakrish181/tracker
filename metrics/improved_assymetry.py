import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# def compute_asymmetry(binary_image):
#     moments = cv2.moments(binary_image)

#     if moments["m00"] == 0:
#         return 0, None, None, None  # Consistent return type

#     cx = int(moments["m10"] / moments["m00"])
#     cy = int(moments["m01"] / moments["m00"])

#     mu20 = moments["mu20"] / moments["m00"]
#     mu02 = moments["mu02"] / moments["m00"]
#     mu11 = moments["mu11"] / moments["m00"]

#     angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

#     rows, cols = binary_image.shape
#     M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1)
#     rotated = cv2.warpAffine(binary_image, M, (cols, rows))

#     flipped = cv2.flip(rotated, 1)
#     diff = cv2.absdiff(rotated, flipped)
#     asymmetry_score = np.sum(diff) / np.sum(rotated)

#     return asymmetry_score, rotated, flipped, diff



#     """
#         There is an issue here! Flipping is not happening on the lesion level, but, on the rotated image! ideally, we need to flip along prinicpal axis of the lesion
#     """





# # Load your binary lesion image (white = lesion, black = background)
# binary_img = cv2.imread("test_images/image_mask.png", cv2.IMREAD_GRAYSCALE)
# #_, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

# score, rotated, flipped, diff = compute_asymmetry(binary_img)

# print(f"Asymmetry Score: {score:.4f}")

# # Show results
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 4, 1)
# plt.title("Original")
# plt.imshow(binary_img, cmap='gray')

# plt.subplot(1, 4, 2)
# plt.title("Rotated")
# plt.imshow(rotated, cmap='gray')

# plt.subplot(1, 4, 3)
# plt.title("Flipped")
# plt.imshow(flipped, cmap='gray')

# plt.subplot(1, 4, 4)
# plt.title("Difference")
# plt.imshow(diff, cmap='hot')
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_asymmetry(original_img, binary_mask):
    # Ensure binary mask is binary (0 and 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Compute moments of the lesion
    moments = cv2.moments(binary_mask)
    if moments["m00"] == 0:
        return 0, None, None, None
    
    # Calculate centroid
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    # Central moments
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    
    # Principal axis angle
    angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
    
    # Get the bounding box of the lesion
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Extract lesion ROI from original image and mask
    lesion_roi = original_img[y:y+h, x:x+w]
    lesion_mask = binary_mask[y:y+h, x:x+w]
    
    # Rotate the lesion ROI
    M = cv2.getRotationMatrix2D((cx - x, cy - y), np.degrees(angle), 1)
    rotated_roi = cv2.warpAffine(lesion_mask, M, (w, h))
    
    # Flip the rotated lesion ROI along the vertical axis (principal axis)
    flipped_roi = cv2.flip(rotated_roi, 1)
    
    # Compute difference within the lesion ROI
    diff = cv2.absdiff(rotated_roi, flipped_roi)
    asymmetry_score = np.sum(diff) / np.sum(rotated_roi) if np.sum(rotated_roi) > 0 else 0
    
    return asymmetry_score, rotated_roi, flipped_roi, diff

# Load original image and binary mask (adjust paths as needed)
original_img = cv2.imread("test_images/image.png", cv2.IMREAD_GRAYSCALE)
binary_mask = cv2.imread("test_images/image_mask.png", cv2.IMREAD_GRAYSCALE)

# Compute asymmetry
score, rotated, flipped, diff = compute_asymmetry(original_img, binary_mask)

print(f"Asymmetry Score: {score:.4f}")

# Show results
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title("Original Mask")
plt.imshow(binary_mask, cmap='gray')

plt.subplot(1, 4, 2)
plt.title("Rotated Lesion")
plt.imshow(rotated, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Flipped Lesion")
plt.imshow(flipped, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Difference")
plt.imshow(diff, cmap='hot')
plt.show()