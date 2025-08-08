from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import cv2
import numpy as np
import os
import pathlib
from match_dir.src.utils.plotting import make_matching_figure
from match_dir.src.loftr import LoFTR, default_cfg
import matplotlib.pyplot as plt
import logging
import shutil
import tempfile

# Fix WindowsPath issue
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'weights/best_1280_default_hyper.pt')

# Load LoFTR model
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# Static output folder
STATIC_DIR = "static/output"
os.makedirs(STATIC_DIR, exist_ok=True)

# Temporary folder for uploaded images
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# FastAPI app initialization
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Pydantic model for request validation
class MatchRequest(BaseModel):
    radius: int = 20  # Matching radius for YOLOv5 centers

@app.get("/")
async def serve_index():
    # Serve index.html directly using FileResponse
    return FileResponse("static/index.html")

@app.post("/match_moles")
async def match_moles(
    image_path1: UploadFile = File(...),
    image_path2: UploadFile = File(...),
    radius: int = Form(20)
):
    try:
        # Step 1: Save uploaded images temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_DIR) as temp_file1:
            shutil.copyfileobj(image_path1.file, temp_file1)
            temp_path1 = temp_file1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_DIR) as temp_file2:
            shutil.copyfileobj(image_path2.file, temp_file2)
            temp_path2 = temp_file2.name

        # Step 2: Load images
        img0_raw = cv2.imread(temp_path1, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(temp_path2, cv2.IMREAD_GRAYSCALE)

        if img0_raw is None or img1_raw is None:
            raise HTTPException(status_code=400, detail="Error reading images.")

        # Resize images for LoFTR
        img0_resized = cv2.resize(img0_raw, (640, 480))
        img1_resized = cv2.resize(img1_raw, (640, 480))

        img0_tensor = torch.from_numpy(img0_resized)[None][None].cuda() / 255.
        img1_tensor = torch.from_numpy(img1_resized)[None][None].cuda() / 255.
        batch = {'image0': img0_tensor, 'image1': img1_tensor}

        # Step 3: Run LoFTR to get keypoints
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Keypoints from img0
            mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Keypoints from img1

        img0_height, img0_width = img0_resized.shape
        img1_height, img1_width = img1_resized.shape

        normalized_mkpts0 = mkpts0 / [img0_width, img0_height]
        normalized_mkpts1 = mkpts1 / [img1_width, img1_height]

        # Step 4: Run YOLOv5 to detect bounding boxes in img0
        results = model(temp_path1)  # YOLO detection
        
        query_points = []
        bboxes = []
        for bbox in results.xyxyn[0].tolist():
            x1, y1, x2, y2 = bbox[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            query_points.append([cx, cy])
            bboxes.append([x1, y1, x2, y2])

        rendered_img = results.render()[0]
        
        query_points = np.array(query_points)  # NumPy array for faster processing

        # Step 5: Find closest LoFTR matches for YOLO centers
        distances = np.linalg.norm(normalized_mkpts0 - query_points[:, None], axis=2)
        closest_idx = np.argmin(distances, axis=1)

        # Step 6: Get matched points
        matched_points = normalized_mkpts1[closest_idx]

        # Step 7: Generate a matching figure
        color = np.random.rand(matched_points.shape[0], 3)  # Random colors for each match
        text = [f'Matches: {len(matched_points)}']

        img0_height, img0_width = img0_raw.shape
        img1_height, img1_width = img1_raw.shape

        re_scale_query = query_points * [img0_width, img0_height]
        re_scale_matched = matched_points * [img1_width, img1_height]

        fig = make_matching_figure(img0_raw, img1_raw, re_scale_query, re_scale_matched, color, re_scale_query, re_scale_matched, text)

        # Step 8: Save and return the result
        annotated_filename = "matched_image.jpg"
        yolo_img = "yolo_image.jpg"
        annotated_path = os.path.join(STATIC_DIR, annotated_filename)
        yolo_img_path = os.path.join(STATIC_DIR, yolo_img)
        plt.savefig(annotated_path)
        plt.close()

        cv2.imwrite(yolo_img_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
        # Clean up temporary files
        os.remove(temp_path1)
        os.remove(temp_path2)

        return JSONResponse(content={
            "message": "Matching complete",
            "image_path": f"/static/output/{annotated_filename}",
            "yolo_image_path": f"/static/output/{yolo_img}",
            "matches_count": len(matched_points),
            "bboxes": bboxes,
            "query_points": query_points.tolist(),
            "matched_points": matched_points.tolist(),
        })

    except Exception as e:
        # Clean up temporary files in case of error
        if 'temp_path1' in locals():
            os.remove(temp_path1)
        if 'temp_path2' in locals():
            os.remove(temp_path2)
        raise HTTPException(status_code=500, detail=str(e))