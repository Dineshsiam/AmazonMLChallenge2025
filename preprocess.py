# preprocess.py
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import DistilBertTokenizer
import re

# IMAGE TRANSFORMS
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(0.15, 0.15, 0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path, transform=val_transform):
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception:
        # fallback zero image (rare)
        return torch.zeros((3, 224, 224), dtype=torch.float32)

# TEXT TOKENIZER
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_SEQ_LEN = 128

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0)
    }

# STRUCTURED FEATURES extractor (same as dataset uses)
def structured_features_from_text(text):
    features = []
    ipq = re.search(r'IPQ[:\s]*(\d+)', text, flags=re.I)
    features.append(int(ipq.group(1)) if ipq else 1)
    text_lower = text.lower()
    features.append(1.0 if any(w in text_lower for w in ['phone', 'laptop', 'tablet', 'camera']) else 0.0)
    features.append(1.0 if any(w in text_lower for w in ['kitchen', 'furniture', 'decor']) else 0.0)
    brand = re.search(r'\b([A-Z][a-z]+)\b', text)
    features.append(float(hash(brand.group(1)) % 1000) if brand else 0.0)
    features.append(float(len(text.split())))
    return torch.tensor(features, dtype=torch.float32)
