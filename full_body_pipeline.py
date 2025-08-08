import torch
import cv2
import os
import numpy as np
from pathlib import Path
from integrated_pipeline import IntegratedMolePipeline
import pathlib
import logging

class FullBodyMoleAnalysisPipeline:
    def __init__(self, yolo_model_path, segmentation_model_path, patch_size=1280, patch_overlap=0.2):
        """
        Initializes the FullBodyMoleAnalysisPipeline.

        Args:
            yolo_model_path (str): Path to the YOLOv5 model weights file.
            segmentation_model_path (str): Path to the segmentation model weights file.
            patch_size (int): Size of each patch for high-resolution image processing.
            patch_overlap (float): Overlap ratio between patches (0-1).
        """
        # Load YOLOv5 model from PyTorch Hub
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
        
        # Initialize the integrated pipeline for segmentation and analysis
        self.integrated_pipeline = IntegratedMolePipeline(segmentation_model_path)
        
        # Set patch parameters
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def create_image_patches(self, image):
        """
        Divides a large image into overlapping patches.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            list: A list of tuples (patch, x_offset, y_offset) containing patch images and their positions.
        """
        h, w = image.shape[:2]
        stride = int(self.patch_size * (1 - self.patch_overlap))  # Stride based on overlap
        patches = []

        # Calculate number of patches in each dimension
        n_h = max(1, int(np.ceil((h - self.patch_size) / stride)) + 1)
        n_w = max(1, int(np.ceil((w - self.patch_size) / stride)) + 1)

        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates
                x_start = min(j * stride, w - self.patch_size)
                y_start = min(i * stride, h - self.patch_size)
                x_end = x_start + self.patch_size
                y_end = y_start + self.patch_size
                
                # Handle edge cases
                if x_end > w:
                    x_end = w
                    x_start = max(0, w - self.patch_size)
                if y_end > h:
                    y_end = h
                    y_start = max(0, h - self.patch_size)
                
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end].copy()
                patches.append((patch, x_start, y_start))
        
        return patches

    def detect_moles(self, image_path):
        """
        Detects moles in a full-body image using the YOLOv5 model.
        For high-resolution images, the image is divided into patches.

        Args:
            image_path (str): Path to the full-body image.

        Returns:
            list: A list of bounding boxes for detected moles.
        """
        # Read the image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Check if the image needs to be processed in patches
        #Reduce the number to make it work in the size.patch_size * thing
        if max(h, w) > self.patch_size * 10:  # If image is significantly larger than patch size
            patches = self.create_image_patches(img)
            all_detections = []
            
            print('Entered patching')
            logging.info('Entered patching')
            for patch, x_offset, y_offset in patches:
                # Save patch to temporary file for YOLO processing
                temp_patch_path = os.path.join(os.path.dirname(image_path), "temp_patch.jpg")
                cv2.imwrite(temp_patch_path, patch)
                
                # Run detection on patch
                results = self.yolo_model(temp_patch_path)
                detections = results.xyxyn[0].cpu().numpy()
                
                # Remove temporary file
                try:
                    os.remove(temp_patch_path)
                except:
                    pass
                
                # Skip if no detections in this patch
                if len(detections) == 0:
                    continue
                
                # Adjust coordinates relative to the full image
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    
                    # Convert normalized coordinates to patch coordinates
                    patch_w, patch_h = patch.shape[1], patch.shape[0]
                    abs_x1 = int(x1 * patch_w) + x_offset
                    abs_y1 = int(y1 * patch_h) + y_offset
                    abs_x2 = int(x2 * patch_w) + x_offset
                    abs_y2 = int(y2 * patch_h) + y_offset
                    
                    # Convert back to normalized coordinates in the full image
                    norm_x1 = abs_x1 / w
                    norm_y1 = abs_y1 / h
                    norm_x2 = abs_x2 / w
                    norm_y2 = abs_y2 / h
                    
                    # Add to all detections
                    all_detections.append([norm_x1, norm_y1, norm_x2, norm_y2, conf, cls])
            
            # Non-maximum suppression to remove duplicate detections
            if all_detections:
                # Convert to numpy array
                all_detections = np.array(all_detections)
                return self.non_max_suppression(all_detections)
            return np.array([])  # Return empty array if no detections
        else:
            # For smaller images, process normally
            results = self.yolo_model(image_path)
            return results.xyxyn[0].cpu().numpy()
    
    def non_max_suppression(self, boxes, iou_threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping bounding boxes.

        Args:
            boxes (numpy.ndarray): Array of detection boxes [x1, y1, x2, y2, confidence, class]
            iou_threshold (float): IoU threshold for considering boxes as duplicates

        Returns:
            numpy.ndarray: Filtered array of detections
        """
        # If no boxes, return empty array
        if len(boxes) == 0:
            return np.array([])
        
        # Initialize list of picked indexes
        pick = []
        
        # Grab coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute area of bounding boxes
        area = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence (use negative confidence for descending order)
        idxs = np.argsort(boxes[:, 4])
        
        # Keep looping while indexes remain in the index list
        while len(idxs) > 0:
            # Grab last index (highest confidence) and add to picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Compute width and height of the intersection
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Compute intersection area
            intersection = w * h
            
            # Compute IoU
            union = area[i] + area[idxs[:last]] - intersection
            iou = intersection / union
            
            # Delete all indexes with IoU greater than threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))
        
        return boxes[pick]

    def crop_moles(self, image_path, detections, output_dir, padding = False):
        """
        Crops detected moles from the original image and saves them.
        Adds padding to ensure all cropped images are 512x512 pixels.

        Args:
            image_path (str): Path to the original full-body image.
            detections (list): A list of bounding box detections.
            output_dir (str): Directory to save the cropped mole images.
            padding (bool): Whether to add padding to the cropped images.

        Returns:
            list: A list of dictionaries, each containing mole_id and cropped_image_path.
        """
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        cropped_moles = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            
            # Convert normalized coordinates to absolute coordinates
            abs_x1 = int(x1 * w)
            abs_y1 = int(y1 * h)
            abs_x2 = int(x2 * w)
            abs_y2 = int(y2 * h)

            # Crop the mole region
            if padding:
                cropped_img = img[abs_y1+25:abs_y2-25, abs_x1+25:abs_x2-25]
                target_size = 512
                pad_top = max(0, (target_size - current_h) // 2)
                pad_bottom = max(0, target_size - current_h - pad_top)
                pad_left = max(0, (target_size - current_w) // 2)
                pad_right = max(0, target_size - current_w - pad_left)

            else:
                cropped_img = img[abs_y1:abs_y2, abs_x1:abs_x2]
                pad_top = 0
                pad_bottom = 0
                pad_left = 0
                pad_right = 0

            # Get current dimensions

            current_h, current_w = cropped_img.shape[:2]


            # If the cropped image is larger than 512x512, resize it to fit
            if current_h > target_size or current_w > target_size:
                # Calculate the scaling factor to fit within 512x512 while preserving aspect ratio
                scale = min(target_size / current_h, target_size / current_w)
                new_h, new_w = int(current_h * scale), int(current_w * scale)
                cropped_img = cv2.resize(cropped_img, (new_w, new_h))
                
                # Recalculate padding after resize
                pad_top = (target_size - new_h) // 2
                pad_bottom = target_size - new_h - pad_top
                pad_left = (target_size - new_w) // 2
                pad_right = target_size - new_w - pad_left
            
            # Add padding to make the image 512x512
            padded_img = cv2.copyMakeBorder(
                cropped_img, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]  # Black padding
            )
            
            # Save the padded image
            mole_id = f"mole_{i+1}"
            original_filename = Path(image_path).stem
            cropped_filename = f"{original_filename}_{mole_id}.png"
            cropped_image_path = os.path.join(output_dir, cropped_filename)
            cv2.imwrite(cropped_image_path, padded_img)
            
            cropped_moles.append({
                'mole_id': mole_id,
                'bbox': [x1, y1, x2, y2],
                'cropped_image_path': cropped_image_path
            })
            
        return cropped_moles

    def process_full_body_image(self, image_path, output_dir):
        """
        Processes a full-body image to detect, crop, and analyze all moles.

        Args:
            image_path (str): Path to the full-body image.
            output_dir (str): Directory to save cropped images and analysis results.

        Returns:
            list: A list of dictionaries, where each dictionary contains the results for a single mole.
        """
        # Step 1: Detect moles in the full-body image
        detections = self.detect_moles(image_path)
        
        if len(detections) == 0:
            return []

        # Step 2: Crop the detected moles and save them
        cropped_moles = self.crop_moles(image_path, detections, output_dir)
        
        # Step 3: Run segmentation and ABCD analysis on each cropped mole
        analysis_results = []
        for mole_info in cropped_moles:
            try:
                # Process each cropped mole image
                abcd_results = self.integrated_pipeline.process_image(
                    mole_info['cropped_image_path'],
                    save_intermediate=True # No need to save intermediate results for each mole here
                )
                
                mole_info['analysis'] = abcd_results
                analysis_results.append(mole_info)
            except Exception as e:
                # Handle cases where analysis might fail for a specific mole
                print(f"Could not analyze mole {mole_info['mole_id']}: {e}")
                mole_info['analysis'] = {'error': str(e)}
                analysis_results.append(mole_info)

        return analysis_results

# Example usage:
if __name__ == '__main__':
    # This is an example of how to use the pipeline. 
    # You would need to provide your own model paths and a test image.
    
    # Define paths
    pathlib.PosixPath = pathlib.WindowsPath
    yolo_model = 'weights/best_1280_default_hyper.pt' # Replace with your YOLO model path
    seg_model = 'weights/segment_mob_unet_.bin' # Replace with your segmentation model path
    test_image = 'test_images/full_body_test.jpg' # Replace with your test image path
    output_directory = 'full_body_output_test'

    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(exist_ok=True)
    
    # Initialize and run the pipeline
    if os.path.exists(yolo_model) and os.path.exists(seg_model) and os.path.exists(test_image):
        pathlib.PosixPath = pathlib.WindowsPath
        full_body_pipeline = FullBodyMoleAnalysisPipeline(
            yolo_model_path=yolo_model, 
            segmentation_model_path=seg_model,
            patch_size=1280,  # Default patch size to match YOLO's expected input
            patch_overlap=0.2  # 20% overlap between patches
        )
        results = full_body_pipeline.process_full_body_image(test_image, output_dir=output_directory)
        
        # Print the results
        for result in results:
            print(f"--- Mole ID: {result['mole_id']} ---")
            print(f"Bounding Box: {result['bbox']}")
            print(f"Cropped Image: {result['cropped_image_path']}")
            if 'error' in result['analysis']:
                print(f"Analysis Error: {result['analysis']['error']}")
            else:
                print(f"ABCD Metrics: {result['analysis']}")
            print("\n")
    else:
        print("Please ensure model weights and test image paths are correct.")
