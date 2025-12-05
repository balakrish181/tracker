# Base image with CUDA + PyTorch preinstalled (choose appropriate tag for your GPU)
FROM runpod/pytorch:2.2.1-py3.10-cuda11.8.0-devel

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenCV, image codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy project
COPY . /workspace

# Vendor YOLOv5 locally to avoid torch.hub network at runtime
RUN git clone --depth=1 https://github.com/ultralytics/yolov5 /opt/yolov5
ENV YOLOV5_DIR=/opt/yolov5

# Install Python dependencies
# Torch/TorchVision already in base image; ensure pip is latest
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /workspace/version1/runpod_worker/requirements.txt \
 && pip install --no-cache-dir -r /opt/yolov5/requirements.txt

# Ensure weights are present (they must be part of the build context)
# Expected at /workspace/version1/weights
ENV PYTHONPATH=/workspace/version1:${PYTHONPATH}

# Entrypoint: start RunPod serverless handler
# Using the handler that calls runpod.serverless.start()
CMD ["python", "-u", "/workspace/version1/runpod_worker/handler.py"]
