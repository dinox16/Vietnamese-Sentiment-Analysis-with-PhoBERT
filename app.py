"""PhoBERT sentiment demo and JSON API in one Hugging Face Space app."""

import os
from contextlib import asynccontextmanager

import gradio as gr
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vsfc.preprocessing import preprocess_text

load_dotenv()

MODEL_ID = os.getenv("HF_MODEL_ID", "Dinox16/phobert-sentiment").strip()
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=False, token=HF_TOKEN
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, token=HF_TOKEN
    )
    model.to(DEVICE).eval()
    yield
    model = None
    tokenizer = None


api = FastAPI(
    title="Vietnamese Student Feedback Sentiment API",
    description="Classify Vietnamese student feedback as negative, neutral, or positive.",
    version="1.0.0",
    lifespan=lifespan,
)


@api.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_ID}


@api.post("/predict")
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
        str(id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))): round(
            float(score), 6
        )
        for index, score in enumerate(probabilities)
    }
    return {
        "text": text,
        "prediction": max(scores, key=scores.get),
        "scores": scores,
    }


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}
    result = predict(PredictionRequest(text=text))
    return result["scores"]


theme = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
).set(
    body_background_fill="#f8f7fc",
    body_background_fill_dark="#11111b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1b1a28",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
)

with gr.Blocks(theme=theme, title="Student Feedback Sentiment") as demo:
    gr.Markdown(
        """
        # Student Feedback Sentiment
        ### Understand the feeling behind a comment.
        Enter a Vietnamese student comment and PhoBERT will estimate whether its tone is **negative**, **neutral**, or **positive**.
        """
    )
    with gr.Row():
        with gr.Column(scale=6):
            comment = gr.Textbox(
                lines=6,
                label="Student comment",
                placeholder="For example: Thầy giảng bài rất dễ hiểu.",
                max_lines=10,
            )
            submit = gr.Button("Analyze sentiment", variant="primary")
            gr.Examples(
                examples=[
                    ["Giảng viên giảng bài rất dễ hiểu."],
                    ["Môn học quá nhàm chán và khó hiểu."],
                    ["Thầy giảng bình thường, không có gì đặc biệt."],
                    ["Thầy dạy hay nhưng không đủ thời gian giải đáp thắc mắc."],
                ],
                inputs=comment,
                label="Try an example",
            )
        with gr.Column(scale=5):
            scores = gr.Label(num_top_classes=3, label="Sentiment scores")
            gr.Markdown(
                """
                **About this model**
                Fine-tuned PhoBERT-base · UIT-VSFC · Vietnamese text

                Scores show the model's confidence for each sentiment class. They are predictions, not a substitute for reading feedback in context.
                """
            )

    submit.click(fn=classify, inputs=comment, outputs=scores)
    comment.submit(fn=classify, inputs=comment, outputs=scores)


app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
