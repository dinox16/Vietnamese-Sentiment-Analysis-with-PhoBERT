
"""Gradio sentiment demo deployed with FastAPI on Vercel."""

import os
import requests
import gradio as gr

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID", "Dinox16/phobert-sentiment"
).strip()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_API_URL = os.getenv(
    "HF_API_URL",
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
).strip()

LABEL_EN = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}

    if not HF_TOKEN:
        raise gr.Error(
            "HF_TOKEN is missing. Configure it in Vercel Environment Variables."
        )

    try:
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            json={"inputs": text.strip()},
            timeout=30,
        )

        response.raise_for_status()
        payload = response.json()

    except requests.Timeout as exc:
        raise gr.Error(
            "Hugging Face API timed out."
        ) from exc

    except requests.HTTPError as exc:
        status = exc.response.status_code
        detail = ""

        try:
            error_payload = exc.response.json()
            if isinstance(error_payload, dict):
                detail = str(
                    error_payload.get(
                        "error",
                        error_payload.get("message", "")
                    )
                )
        except ValueError:
            pass

        if status in (401, 403):
            message = "Invalid HF_TOKEN or insufficient permissions."
        elif status == 404:
            message = "Model inference endpoint not found."
        elif status == 429:
            message = "Hugging Face rate limit exceeded."
        elif status == 503:
            message = "Inference service temporarily unavailable."
        else:
            message = f"Hugging Face API error: {status}"

        if detail:
            message += f" {detail}"

        raise gr.Error(message) from exc

    except requests.RequestException as exc:
        raise gr.Error(
            "Failed to connect to Hugging Face."
        ) from exc

    except ValueError as exc:
        raise gr.Error(
            "Hugging Face returned invalid JSON."
        ) from exc

    # Accept either flat or nested classification output.
    if isinstance(payload, list) and payload:
        predictions = (
            payload[0]
            if isinstance(payload[0], list)
            else payload
        )
    else:
        predictions = payload

    if not isinstance(predictions, list):
        raise gr.Error(
            "Unexpected response format. Check model inference support."
        )

    scores = {}

    for item in predictions:
        if not isinstance(item, dict):
            continue

        if "label" not in item or "score" not in item:
            continue

        label = str(item["label"])
        display_label = LABEL_EN.get(
            label.lower(), label
        )

        scores[display_label] = float(item["score"])

    if not scores:
        raise gr.Error(
            "No sentiment predictions were returned."
        )

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


with gr.Blocks(
    title="Student Feedback Sentiment"
) as demo:

    gr.Markdown(
        """
        # Student Feedback Sentiment
        ### Understand the feeling behind a comment.

        Enter a Vietnamese student comment and PhoBERT
        will estimate whether its tone is **negative**,
        **neutral**, or **positive**.
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

            submit = gr.Button(
                "Analyze sentiment",
                variant="primary"
            )
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

            scores = gr.Label(
                num_top_classes=3,
                label="Sentiment scores"
            )

            gr.Markdown(
                """
                **About this model**

                Fine-tuned PhoBERT-base · UIT-VSFC · Vietnamese text

                Scores show the model's confidence for each
                sentiment class. They are predictions, not a
                substitute for reading feedback in context.
                """
            )

    submit.click(
        fn=classify,
        inputs=comment,
        outputs=scores,
    )

    comment.submit(
        fn=classify,
        inputs=comment,
        outputs=scores,
    )


app = FastAPI()

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=theme,
)


# Local development only.
if __name__ == "__main__":
    demo.launch(theme=theme)
