
# AI Text Detection Pipeline

A high-performance, dual-backend solution for AI-based text detection, supporting both PyTorch and TensorRT inference engines. This project provides a FastAPI deployment with Docker support for easy implementation and benchmarking.

## 🚀 Key Features

- **Dual Backend Support**: Switch between PyTorch and TensorRT inference
- **High Performance**: Optimized TensorRT FP16 model for production deployment
- **RESTful API**: FastAPI endpoint for easy integration
- **Dockerized**: Complete containerization with GPU support
- **Benchmarking Tools**: Performance comparison utilities

## 📊 Performance Highlights

| Metric | PyTorch FP32 | TensorRT FP16 | Improvement |
|--------|--------------|---------------|-------------|
| Total Inference Time | 277.53 sec | 141.41 sec | ~1.96× faster |
| Avg. Latency per Sample | 0.648 sec | 0.330 sec | ~1.96× faster |
| Throughput | 7.21 samples/sec | 14.14 samples/sec | ~1.96× higher |
| GPU Memory Usage | ~3.4 GB | ~0.0 GB | Pre-allocated |
| Accuracy | 0.8025 | 0.8005 | Negligible difference |
| F1-Score | 0.795 | 0.793 | Negligible difference |

## 📁 Project Structure
---

ai_textdetection_pipeline/
├── models/
│ └── optimized/
│ ├── desklib.onnx # ONNX model
│ └── desklib_trt_fp16_64.engine # TensorRT optimized engine
├── src/
│ ├── api/
│ │ ├── init.py
│ │ ├── server_trt.py # TensorRT FastAPI server
│ │ └── server_pytorch.py # PyTorch FastAPI server
│ ├── pipeline/
│ │ ├── init.py
│ │ ├── engine_builder.py # TRT engine construction
│ │ ├── preprocess.py # Input preprocessing
│ │ └── inference.py # Core inference logic
│ └── run_app.py # Application launcher
├── benchmarks/ # Benchmarking scripts (optional)
├── tests/ # Test cases (optional)
├── Dockerfile
├── requirements.txt
├── docker-compose.yml # Added for multi-container setup
└── README.md

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Chhalma/ai_textdetection_optimized_pipeline.git
cd ai_textdetection_pipeline

### 2. Install Python dependencies
pip install -r requirements.txt

### 3. Run the API locally

#TensorRT backend:

python src/run_app.py --backend tensorrt

#PyTorch backend:

python src/run_app.py --backend pytorch


#The API will be available at http://127.0.0.1:8000.

Using Docker
1. Build the Docker image
docker build -t ai_textdetection_pipeline .

2. Run the container (TensorRT backend by default)
docker run --gpus all -p 8000:8000 ai_textdetection_pipeline

3. Run with PyTorch backend
docker run --gpus all -p 8000:8000 -e BACKEND=pytorch ai_textdetection_pipeline


⚠️ Note: A GPU is required to run the Docker container with TensorRT or PyTorch backends.

Testing the API

Once the server is running, you can test it using curl or Postman.

Example curl request:

curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@path_to_sample_image.jpg"

The API returns the detection results in JSON format.

Replace path_to_sample_image.jpg with your local image path.

Notes

Models are located in models/optimized/. Make sure both .onnx and .engine files are present.

run_app.py supports backend selection (--backend pytorch or --backend tensorrt).

Docker setup uses NVIDIA’s PyTorch container image, so GPU is required for inference.
