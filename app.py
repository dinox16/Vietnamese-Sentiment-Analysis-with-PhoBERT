"""Single-file Gradio frontend / Hugging Face inference app.

Set APP_MODE=space on Hugging Face Spaces to load PhoBERT and serve /predict.
The default Vercel mode only calls the public Space API and stays lightweight.
"""

import os

import gradio as gr
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

MODEL_ID = os.getenv("HF_MODEL_ID", "Dinox16/phobert-sentiment").strip()
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
SPACE_API_URL = os.getenv(
    "SPACE_API_URL", "https://dinox16-pho-bert-dtu.hf.space/predict"
).strip()
SPACE_API_TOKEN = os.getenv("SPACE_API_TOKEN", "").strip()
APP_MODE = os.getenv("APP_MODE", "vercel").strip().lower()

tokenizer = None
model = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


api = FastAPI(title="Vietnamese Student Feedback Sentiment API")


if APP_MODE == "space":
    import torch
    from contextlib import asynccontextmanager
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from vsfc.preprocessing import preprocess_text

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
        model.to(DEVICE).eval()
        yield
        model = None
        tokenizer = None

    api.router.lifespan_context = lifespan

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
            preprocess_text(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()
        id2label = model.config.id2label
        scores = {
            str(id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))): round(float(score), 6)
            for index, score in enumerate(probabilities)
        }
        return {"text": text, "prediction": max(scores, key=scores.get), "scores": scores}

else:
    @api.get("/health")
    def health_check():
        return {"status": "ok", "mode": "vercel-proxy", "space_api": SPACE_API_URL}

    @api.post("/predict")
    def predict(request: PredictionRequest):
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="Text must not be blank.")
        headers = {"Content-Type": "application/json"}
        if SPACE_API_TOKEN:
            headers["Authorization"] = f"Bearer {SPACE_API_TOKEN}"
        try:
            response = requests.post(
                SPACE_API_URL,
                json={"text": request.text.strip()},
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout as exc:
            raise HTTPException(status_code=504, detail="Hugging Face Space timed out.") from exc
        except requests.HTTPError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail="Hugging Face Space returned an error.",
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise HTTPException(status_code=502, detail="Could not get a valid response from the Space.") from exc


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        response = requests.post(
            SPACE_API_URL,
            json={"text": text.strip()},
            headers={"Authorization": f"Bearer {SPACE_API_TOKEN}"} if SPACE_API_TOKEN else {},
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()
    except requests.Timeout as exc:
        raise gr.Error("The Hugging Face Space took too long to respond.") from exc
    except requests.HTTPError as exc:
        raise gr.Error(f"The Hugging Face Space returned error {exc.response.status_code}.") from exc
    except (requests.RequestException, ValueError) as exc:
        raise gr.Error("Could not get a valid response from the Hugging Face Space.") from exc
    returned_scores = result.get("scores") if isinstance(result, dict) else None
    if not isinstance(returned_scores, dict):
        raise gr.Error("The Hugging Face Space returned an unexpected response.")
    return returned_scores


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

with gr.Blocks(title="Student Feedback Sentiment") as demo:
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
            gr.Markdown(
                """
                **Example comments**

                - Giảng viên giảng bài rất dễ hiểu.
                - Môn học quá nhàm chán và khó hiểu.
                - Thầy giảng bình thường, không có gì đặc biệt.
                - Thầy dạy hay nhưng không đủ thời gian giải đáp thắc mắc.
                """
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


app = gr.mount_gradio_app(api, demo, path="/", theme=theme)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
