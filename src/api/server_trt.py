
# api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.preprocess import preprocess, preprocess_batch

ENGINE_PATH = "/content/drive/MyDrive/ai_textdetection_pipeline/models/optimized/desklib_trt_fp16_64.engine"

app = FastAPI(title="AI Text Detection API (TensorRT Only)")

trt_engine = None

class SingleTextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: list[str]

# === STARTUP EVENT: LOAD TENSORRT ENGINE ===
@app.on_event("startup")
async def load_trt_engine():
    global trt_engine
    from pipeline.inference_tensorrt import InferenceTensorRT

    print("🔧 Loading TensorRT engine on startup...")
    trt_engine = InferenceTensorRT(ENGINE_PATH, max_batch=64, seq_len=768)
    print("✅ TensorRT engine loaded successfully")

# === SHUTDOWN EVENT: CLEANUP ENGINE ===
@app.on_event("shutdown")
async def cleanup_trt_engine():
    global trt_engine
    if trt_engine:
        trt_engine.cleanup()
    print("🧹 TensorRT engine cleaned up on shutdown")

@app.get("/")
async def root():
    return {"message": "AI Text Detection API (TensorRT) is running!"}

@app.post("/single")
async def infer_single(input_data: SingleTextInput):
    text = input_data.text
    input_ids, attention_mask = preprocess(text)
    pred = trt_engine.run(input_ids.numpy(), attention_mask.numpy())
    return {"prediction": pred}

@app.post("/batch")
async def infer_batch(input_data: BatchTextInput):
    texts = input_data.texts
    input_ids, attention_mask = preprocess_batch(texts)
    preds = trt_engine.run(input_ids.numpy(), attention_mask.numpy())
    return {"predictions": preds}
