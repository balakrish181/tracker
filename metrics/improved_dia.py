import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def longest_distance_calculation(binary_mask):
    # Ensure binary mask is binary (0 and 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the lesion
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # Convert binary mask to color for visualization
    mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the longest distance line on the mask
    if pt1 is not None and pt2 is not None:
        cv2.line(mask_color, tuple(pt1), tuple(pt2), (0, 0, 255), 2)  # Red line
        cv2.circle(mask_color, tuple(pt1), 5, (0, 255, 0), -1)  # Green point
        cv2.circle(mask_color, tuple(pt2), 5, (0, 255, 0), -1)  # Green point

    return max_distance, mask_color, pt1, pt2

# Load the binary mask of the lesion (adjust path as needed)
binary_mask = cv2.imread("test_images/image_mask.png", cv2.IMREAD_GRAYSCALE)

# Check if the mask loaded successfully
if binary_mask is None:
    raise ValueError("Binary mask not loaded. Check the file path.")

# Calculate the longest distance
max_distance, mask_with_line, pt1, pt2 = longest_distance_calculation(binary_mask)

# Convert distance to millimeters (assuming 1 pixel = 0.1 mm; adjust based on your image scale)
pixel_to_mm = 0.1  # Example scaling factor; adjust based on your image resolution
max_distance_mm = max_distance * pixel_to_mm

print(f"Longest Distance (pixels): {max_distance:.2f}")
print(f"Longest Distance (mm): {max_distance_mm:.2f}")

# Visualize the result
plt.figure(figsize=(6, 4))
plt.title("Lesion with Longest Distance")
plt.imshow(cv2.cvtColor(mask_with_line, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()