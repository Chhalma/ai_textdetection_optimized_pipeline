
import torch
from transformers import AutoTokenizer, AutoConfig
from pipeline.desklib import DesklibAIDetectionModel

class InferenceBaseModel:
    """Inference for PyTorch-based AI detection models"""

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("desklib/ai-text-detector-academic-v1.01")

        # Instantiate model
        config = AutoConfig.from_pretrained("desklib/ai-text-detector-academic-v1.01")
        self.model = DesklibAIDetectionModel(config)

        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")  # load to CPU first
        self.model.load_state_dict(state_dict)                   # load state_dict into model

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        print("✅ PyTorch base model loaded successfully.")

    def run(self, input_data: str):
        enc = self.tokenizer(input_data, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
        return probs

    def run_batch(self, input_list):
        return [self.run(text) for text in input_list]

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
