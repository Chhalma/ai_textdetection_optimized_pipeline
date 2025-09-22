

# api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.preprocess import preprocess, preprocess_batch

TORCH_MODEL_PATH = "/content/drive/MyDrive/ai_textdetection_pipeline/models/optimized/desklib.pt"

app = FastAPI(title="AI Text Detection API (Torch Only)")

torch_model = None

class SingleTextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: list[str]

# === STARTUP EVENT: LOAD TORCH MODEL ===
@app.on_event("startup")
async def load_model():
    global torch_model
    from pipeline.inference_base_model import InferenceBaseModel

    print("🔧 Loading Torch model on startup...")
    torch_model = InferenceBaseModel(TORCH_MODEL_PATH)
    print("✅ Torch model loaded successfully")

# === SHUTDOWN EVENT: CLEANUP MODEL ===
@app.on_event("shutdown")
async def cleanup_model():
    global torch_model
    if torch_model:
        torch_model.cleanup()
    print("🧹 Torch model cleaned up on shutdown")

@app.get("/")
async def root():
    return {"message": "AI Text Detection API (Torch) is running!"}

@app.post("/single")
async def infer_single(input_data: SingleTextInput):
    text = input_data.text
    pred = torch_model.run(text)
    return {"prediction": pred, "backend": "torch"}

@app.post("/batch")
async def infer_batch(input_data: BatchTextInput):
    texts = input_data.texts
    preds = torch_model.run_batch(texts)
    return {"predictions": preds, "backend": "torch"}
