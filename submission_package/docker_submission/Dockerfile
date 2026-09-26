# BraTS 2021 Winning Architecture - Docker Submission
# Author: Tarek Al Ahmadieh (Sole Author & Developer)
# Target Platform: Synapse.org / MedPerf / Docker Hub

FROM python:3.10-slim

# Avoid prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and core scientific imaging stack
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    nibabel \
    monai \
    pyyaml

# Copy inference engine and model weights
COPY run_inference.py /workspace/run_inference.py

# Create input and output mount points standard for BraTS challenges
RUN mkdir -p /input /output

# Set entrypoint to run inference on input volume
ENTRYPOINT ["python", "/workspace/run_inference.py", "--input_dir", "/input", "--output_dir", "/output"]
