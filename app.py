from flask import Flask, render_template, request, jsonify, send_from_directory
from flask.json.provider import JSONProvider
import json
import os
from pathlib import Path
from integrated_pipeline import IntegratedMolePipeline
from full_body_pipeline import FullBodyMoleAnalysisPipeline
import cv2
import numpy as np
from datetime import datetime
import pathlib

# Monkey patch for Windows to handle PosixPath issue with YOLO model loading
import platform
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Custom JSON provider to handle numpy types
class CustomJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, **kwargs, cls=NumpyJSONEncoder)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

app = Flask(__name__)
app.json = CustomJSONProvider(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
FULL_BODY_OUTPUT_FOLDER = 'full_body_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['FULL_BODY_OUTPUT_FOLDER'] = FULL_BODY_OUTPUT_FOLDER

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
Path(FULL_BODY_OUTPUT_FOLDER).mkdir(exist_ok=True)

# Initialize the pipelines
pipeline = IntegratedMolePipeline()
full_body_pipeline = FullBodyMoleAnalysisPipeline(
    yolo_model_path='weights/best_1280_default_hyper.pt', 
    segmentation_model_path='weights/segment_mob_unet_.bin'
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_full_body', methods=['POST'])
def analyze_full_body():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = full_body_pipeline.process_full_body_image(
                filepath, 
                output_dir=app.config['FULL_BODY_OUTPUT_FOLDER']
            )
            
            # Convert paths for web access and prepare response
            response_results = []
            for result in results:
                if 'cropped_image_path' in result:
                    result['cropped_image_path'] = f"/{app.config['FULL_BODY_OUTPUT_FOLDER']}/{Path(result['cropped_image_path']).name}"
                    response_results.append(result)
            
            # Get the full image dimensions for reference
            img = cv2.imread(filepath)
            if img is not None:
                height, width = img.shape[:2]
            else:
                width, height = 0, 0
                
            return jsonify({
                'results': response_results, 
                'original_image': f'/uploads/{filename}',
                'image_dimensions': {'width': width, 'height': height}
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            results = pipeline.process_image(filepath, save_intermediate=True, output_dir=app.config['OUTPUT_FOLDER'])
            
            # Prepare response data
            response_data = {
                'metrics': results,
                'original_image': f'/uploads/{filename}',
                'mask_image': f'/outputs/{Path(filename).stem}_mask.png',
                'overlay_image': f'/outputs/{Path(filename).stem}_overlay.png'
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/full_body_output/<filename>')
def full_body_output_file(filename):
    return send_from_directory(app.config['FULL_BODY_OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port = 5001) 