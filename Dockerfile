# Clone model
FROM alpine/git:2.47.2 AS clone
COPY clone.sh /clone.sh
RUN . /clone.sh /workspace/models/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 main

# Build final image
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
    pip install vllm runpod

# Copy model from clone stage
COPY --from=clone /workspace/models /workspace/models

# Copy handler
WORKDIR /src
COPY src/handler.py .

# Set default environment variables
ENV VLLM_MODEL=/workspace/models/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 \
    VLLM_TRUST_REMOTE_CODE=true \
    VLLM_ENFORCE_EAGER=true \
    VLLM_QUANTIZATION=compressed-tensors \
    VLLM_MAX_MODEL_LEN=8192

CMD ["python3", "handler.py"]