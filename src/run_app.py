
import uvicorn
import argparse

def main(backend: str, host: str, port: int):
    if backend == "pytorch":
        app_module = "api.server_pytorch:app"
    elif backend == "tensorrt":
        app_module = "api.server_trt:app"
    else:
        raise ValueError("Invalid backend. Choose 'pytorch' or 'tensorrt'.")
    
    uvicorn.run(app_module, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument("--backend", type=str, default="pytorch", help="Backend: pytorch or tensorrt")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    main(args.backend, args.host, args.port)
