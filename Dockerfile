# Use a Runpod base image with PyTorch and CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Agree to Coqui TTS Terms of Service to allow model downloading
ENV COQUI_TOS_AGREED=1

# Set working directory
WORKDIR /app

# Install system dependencies like ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the handler file
COPY handler.py .

# Set the command to run the worker
CMD ["python", "-u", "handler.py"]
