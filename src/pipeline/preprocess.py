
from transformers import AutoTokenizer

# Load tokenizer directly from Hugging Face hub
TOKENIZER_NAME = "desklib/ai-text-detector-academic-v1.01"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def preprocess(text: str, max_length: int = 768):
    enc = tokenizer(text, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

def preprocess_batch(texts: list[str], max_length: int = 768):
    enc = tokenizer(texts, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]
