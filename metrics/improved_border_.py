import cv2
import numpy as np
import matplotlib.pyplot as plt

def border_irregularity_index(binary_mask):
    # Ensure binary mask is binary (0 and 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the lesion
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    # Get the largest contour (assumed to be the lesion)
    contour = max(contours, key=cv2.contourArea)

    # Calculate perimeter and area of the lesion
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    # Calculate the radius of a circle with the same area
    radius = np.sqrt(area / np.pi)

    # Perimeter of a circle with the same area
    circle_perimeter = 2 * np.pi * radius

    # Border Irregularity Index: ratio of actual perimeter to circle perimeter
    irregularity_index = perimeter / circle_perimeter if circle_perimeter > 0 else 0

    return irregularity_index

def fractal_dimension(binary_mask, box_sizes=None):
    # Ensure binary mask is binary (0 and 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Default box sizes for box-counting method
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    # Initialize counts for each box size
    counts = []

    for size in box_sizes:
        # Count the number of boxes that contain part of the lesion
        count = 0
        for i in range(0, binary_mask.shape[0], size):
            for j in range(0, binary_mask.shape[1], size):
                if np.any(binary_mask[i:i+size, j:j+size] > 0):
                    count += 1
        counts.append(count)

    # Fit a line to log(box_sizes) vs log(counts) to estimate fractal dimension
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(counts))

    # Linear regression to find the slope (fractal dimension)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]

    return fractal_dim

# Load the binary mask of the lesion (adjust path as needed)
binary_mask = cv2.imread("test_images/image_mask.png", cv2.IMREAD_GRAYSCALE)
print(np.unique(binary_mask)      )

# Calculate Border Irregularity Index
irregularity_index = border_irregularity_index(binary_mask)
print(f"Border Irregularity Index: {irregularity_index:.4f}")

# Calculate Fractal Dimension
fractal_dim = fractal_dimension(binary_mask)
print(f"Fractal Dimension: {fractal_dim:.4f}")

# Visualize the binary mask
plt.figure(figsize=(6, 4))
plt.title("Lesion Binary Mask")
plt.imshow(binary_mask, cmap='gray')
plt.show()