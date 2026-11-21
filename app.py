"""Gradio demo that calls a sentiment model hosted on Hugging Face."""

import os
from fastapi import FastAPI
import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Dinox16/phobert-sentiment").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"

LABEL_EN = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}
    if not HF_TOKEN:
        raise gr.Error("HF_TOKEN is missing. Add your Hugging Face token to the .env file.")

    try:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text.strip()},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise gr.Error("The Hugging Face API took too long to respond. Please try again.") from exc
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except (ValueError, AttributeError):
            pass
        if response.status_code in (401, 403):
            message = "Hugging Face rejected the token. Check HF_TOKEN and its permissions."
        elif response.status_code == 503:
            message = "The model is loading on Hugging Face. Please try again shortly."
        else:
            message = f"Hugging Face API request failed ({response.status_code})."
        if detail:
            message += f" {detail}"
        raise gr.Error(message) from exc
    except requests.RequestException as exc:
        raise gr.Error("Could not connect to the Hugging Face API. Check your connection and model ID.") from exc

    # Text classification endpoints normally return [[{label, score}, ...]].
    # Accept a flat list as well for compatible inference response formats.
    predictions = payload[0] if payload and isinstance(payload[0], list) else payload
    if not isinstance(predictions, list):
        raise gr.Error("The model returned an unexpected response. Check that it supports text classification.")

    scores = {}
    for item in predictions:
        if not isinstance(item, dict) or "label" not in item or "score" not in item:
            continue
        label = str(item["label"])
        display_label = LABEL_EN.get(label.lower(), label)
        scores[display_label] = float(item["score"])
    if not scores:
        raise gr.Error("The model response did not contain sentiment labels and scores.")
    return scores


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
app = FastAPI()
app = gr.mount_gradio_app(
        app,
        demo,
        path="/"
    )


if __name__ == "__main__":
    demo.launch()
