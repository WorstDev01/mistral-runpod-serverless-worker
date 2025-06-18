FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCHDYNAMO_DISABLE=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install vllm runpod huggingface-hub

# Download model using huggingface-hub (more efficient than git)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8', \
    local_dir='/workspace/models/Mistral-Small' \
)"

# Copy handler
WORKDIR /src
COPY src/handler.py .

CMD ["python3", "handler.py"]