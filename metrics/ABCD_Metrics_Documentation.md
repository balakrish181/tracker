# ABCD Metrics for Melanoma Detection

This documentation explains the improved methods used to calculate the ABCD metrics (Asymmetry, Border, Color, Diameter) for melanoma detection in the Mole-mapping application.

## Overview

The ABCD rule is a clinical diagnostic method used to evaluate the malignancy potential of skin lesions, particularly melanoma. It stands for:

- **A**: Asymmetry
- **B**: Border irregularity
- **C**: Color variegation
- **D**: Diameter

Each of these features is evaluated quantitatively using computer vision techniques in our application, resulting in a score that can help assess the risk of malignancy.

## Implementation Details

### 1. Asymmetry Calculation

**Method**: `compute_asymmetry`

**Description**:  
This method measures asymmetry by rotating the lesion mask 180 degrees around its centroid and calculating the mismatch between the original and rotated masks. The more asymmetric a lesion is, the less overlap there will be after rotation.

**Formula**:  
```
Asymmetry = 1 - (Area of Intersection / Area of Union)
```

**Score Interpretation**:
- 0 = perfectly symmetric (original and rotated masks completely overlap)
- 1 = completely asymmetric (no overlap between original and rotated masks)

**Implementation Details**:
1. Find the centroid of the lesion using moments
2. Rotate the mask 180 degrees around the centroid
3. Calculate the intersection and union of the original and rotated masks
4. Compute the asymmetry score using the formula above

**Advantages over Traditional Methods**:
- More accurate than simple geometric asymmetry measures
- Takes into account the entire shape of the lesion
- Considers asymmetry along all axes, not just predetermined ones

### 2. Border Irregularity Calculation

**Method**: `border_irregularity_index`

**Description**:  
This method quantifies border irregularity by comparing the actual perimeter of the lesion contour to the perimeter of a perfect circle with the same area. The more irregular the border, the higher the ratio will be.

**Formula**:  
```
Border Irregularity = Actual Perimeter / (2 * π * √(Area/π))
```
Where (2 * π * √(Area/π)) is the perimeter of a circle with the same area.

**Score Interpretation**:
- 1.0 = perfectly circular border (minimal irregularity)
- >1.0 = increasing border irregularity (more jagged/irregular border)

**Implementation Details**:
1. Find the contours of the lesion
2. Calculate the perimeter and area of the lesion
3. Calculate the radius of a circle with the same area
4. Calculate the perimeter of this theoretical circle
5. Compute the irregularity index as the ratio of actual perimeter to circle perimeter

**Advantages over Traditional Methods**:
- More robust to scale changes
- Provides a normalized measure regardless of lesion size
- Based on well-established geometric principles

### 3. Diameter Calculation

**Method**: `calculate_diameter`

**Description**:  
This method determines the maximum diameter of the lesion by finding the longest distance between any two points on the lesion contour (Feret diameter or maximum caliper distance).

**Implementation Details**:
1. Extract all points from the lesion contour
2. Calculate distances between all pairs of points
3. Find the maximum distance
4. Optional: Convert pixels to millimeters using a calibration factor

**Advantages over Traditional Methods**:
- More accurate than simple width or height measurements
- Finds the true maximum dimension regardless of orientation
- Not limited to horizontal/vertical measurements
- Better aligns with clinical assessment techniques

### 4. Color Analysis

**Method**: `color_space_analysis`

**Description**:  
This method analyzes color variation within the lesion by converting the image to the L*a*b* color space and calculating the variance across all three channels for lesion pixels only.

**Implementation Details**:
1. Convert the lesion image to L*a*b* color space
2. Apply the mask to focus on lesion pixels only
3. Calculate variance for each channel (L*, a*, b*)
4. Sum the variances to get a total color variance score

**Advantages over Traditional Methods**:
- Uses L*a*b* color space which better represents human color perception
- Considers all components of color (lightness, green-red, blue-yellow)
- Focuses only on pixels within the lesion
- More comprehensive than using only hue from HSV space

## Usage in the Application

These metrics are calculated in the `analyze` method of the `MoleAnalyzer` class. The returned scores are normalized to maintain consistency with the original scoring system:

```python
return {
    "Asymmetry": Asymmetry_py,  # Scaled to 0-10 range
    "Border": Border_py,        # Scaled by 1/10 
    "Diameter": Diameter_py,    # Scaled by 1/10
    "Colour": Colour_py,        # Scaled appropriately
    "Raw_Metrics": {
        "Area_pixels": A_py,
        "Asymmetry_0_1": asymmetry_raw_py,
        "Border_CircularityIndex": border_raw_py,
        "Diameter_Feret_pixels": diameter_raw_py
    }
}
```

## Scientific Background

The ABCD rule was first introduced by Nachbar et al. in 1994 as a clinical tool for early melanoma detection. The computational implementation of these metrics allows for more objective assessment but should always be used in conjunction with clinical evaluation by healthcare professionals.

## References

1. Nachbar, F., Stolz, W., Merkle, T., Cognetta, A. B., Vogt, T., Landthaler, M., ... & Plewig, G. (1994). The ABCD rule of dermatoscopy: high prospective value in the diagnosis of doubtful melanocytic skin lesions. Journal of the American Academy of Dermatology, 30(4), 551-559.

2. Celebi, M. E., Kingravi, H. A., Uddin, B., Iyatomi, H., Aslandogan, Y. A., Stoecker, W. V., & Moss, R. H. (2007). A methodological approach to the classification of dermoscopy images. Computerized Medical Imaging and Graphics, 31(6), 362-373.

3. Seidenari, S., Pellacani, G., & Grana, C. (2005). Colors in atypical nevi: a computer description reproducing clinical assessment. Skin Research and Technology, 11(1), 36-41.
