"""Small FastAPI inference service for deployment as a Hugging Face Docker Space."""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vsfc.preprocessing import preprocess_text

MODEL_ID = os.getenv("HF_MODEL_ID", "Dinox16/phobert-sentiment")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model.to(DEVICE).eval()
    yield
    model = None
    tokenizer = None


app = FastAPI(
    title="Vietnamese Student Feedback Sentiment API",
    description="Classify Vietnamese student feedback as negative, neutral, or positive.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict")
@torch.inference_mode()
def predict(request: PredictionRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be blank.")

    inputs = tokenizer(
        preprocess_text(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)
    probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()
    id2label = model.config.id2label
    scores = {
        str(id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))): round(float(score), 6)
        for index, score in enumerate(probabilities)
    }
    prediction = max(scores, key=scores.get)
    return {"text": text, "prediction": prediction, "scores": scores}
