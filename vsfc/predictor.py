"""Nạp model đã fine-tune và dự đoán cảm xúc."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import preprocess_text


class SentimentPredictor:
    def __init__(self, model_name_or_path: str, device: str | None = None, max_length: int = 128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        # PhoBERT chỉ có tokenizer dạng slow (BPE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, texts: str | list[str]) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            [preprocess_text(t) for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        probs = F.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()

        return [
            {
                "text": text,
                "prediction": self.id2label[max(range(len(p)), key=p.__getitem__)],
                "probabilities": {self.id2label[i]: round(v, 4) for i, v in enumerate(p)},
            }
            for text, p in zip(texts, probs)
        ]
