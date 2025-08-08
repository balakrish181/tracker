import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_space_analysis(original_img, binary_mask):
    # Ensure binary mask is binary (0 and 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert mask to 3 channels to match the original image
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply mask to isolate the lesion
    lesion_img = cv2.bitwise_and(original_img, mask_3ch)
    
    # Convert the lesion image to RGB
    img_rgb = cv2.cvtColor(lesion_img, cv2.COLOR_BGR2RGB)
    
    # Convert to CIE L*a*b*
    img_lab = cv2.cvtColor(lesion_img, cv2.COLOR_BGR2LAB)
    
    # Apply mask to L*a*b* image to focus on lesion pixels only
    mask_binary = binary_mask // 255  # Convert to 0 and 1
    mask_expanded = np.expand_dims(mask_binary, axis=2)  # Shape to (H, W, 1)
    mask_expanded = np.repeat(mask_expanded, 3, axis=2)  # Shape to (H, W, 3)
    img_lab_masked = img_lab * mask_expanded  # Zero out non-lesion pixels
    
    # Calculate variance for lesion pixels only
    lesion_pixels = img_lab_masked[mask_expanded == 1].reshape(-1, 3)
    if len(lesion_pixels) == 0:
        return 0, img_rgb, img_lab
    
    lab_variances = np.var(lesion_pixels, axis=0)
    l_variance, a_variance, b_variance = lab_variances
    
    # Total color variance
    total_variance = np.sum(lab_variances)
    
    # Visualize the lesion and L*a*b* channels
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.title("Lesion Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("L* Channel")
    plt.imshow(img_lab[:, :, 0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("a* Channel")
    plt.imshow(img_lab[:, :, 1], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("b* Channel")
    plt.imshow(img_lab[:, :, 2], cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    print(f"L* Variance: {l_variance:.4f}")
    print(f"a* Variance: {a_variance:.4f}")
    print(f"b* Variance: {b_variance:.4f}")
    print(f"Total Color Variance: {total_variance:.4f}")
    
    return total_variance, img_rgb, img_lab

# Load the original image and binary mask (adjust paths as needed)
original_img = cv2.imread("test_images/image.png")
binary_mask = cv2.imread("test_images/image_mask.png", cv2.IMREAD_GRAYSCALE)

# Check if images loaded successfully
if original_img is None or binary_mask is None:
    raise ValueError("Image or mask not loaded. Check the file paths.")

# Calculate color variance for the lesion
total_variance, _, _ = color_space_analysis(original_img, binary_mask)