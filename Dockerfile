
# ===== Base Image with GPU support =====
FROM nvcr.io/nvidia/pytorch:23.10-py3

# ===== Set Working Directory =====
WORKDIR /app

# ===== Copy requirements first (for caching) =====
COPY requirements.txt .

# ===== Install Python Dependencies =====
RUN pip install --no-cache-dir -r requirements.txt

# ===== Copy source code =====
COPY src/ ./src

# ===== Copy models =====
COPY models/ ./models

# ===== Expose FastAPI port =====
EXPOSE 8000

# ===== Set environment variables for CUDA & TensorRT =====
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV PYTHONUNBUFFERED=1


# ===== Start FastAPI (TensorRT server) =====
CMD ["uvicorn", "src.api.server_trt:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
