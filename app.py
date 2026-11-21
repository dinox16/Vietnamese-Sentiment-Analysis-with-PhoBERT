
"""Gradio sentiment demo deployed with FastAPI on Vercel."""

import os
import requests
import gradio as gr

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

SENTIMENT_API_URL = os.getenv(
    "SENTIMENT_API_URL",
    "https://dinox16-phobert-sentiment-api.hf.space/predict",
).strip()
SENTIMENT_API_TOKEN = os.getenv("SENTIMENT_API_TOKEN", "").strip()


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}

    try:
        headers = {"Content-Type": "application/json"}
        if SENTIMENT_API_TOKEN:
            headers["Authorization"] = f"Bearer {SENTIMENT_API_TOKEN}"
        response = requests.post(
            SENTIMENT_API_URL,
            headers=headers,
            json={"text": text.strip()},
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
            message = "The sentiment API rejected the request. Check the Space access token."
        elif status == 404:
            message = "Model inference endpoint not found."
        elif status == 429:
            message = "Hugging Face rate limit exceeded."
        elif status == 503:
            message = "Inference service temporarily unavailable."
        else:
            message = f"Sentiment API error: {status}"

        if detail:
            message += f" {detail}"

        raise gr.Error(message) from exc

    except requests.RequestException as exc:
        raise gr.Error(
            "Failed to connect to the sentiment API."
        ) from exc

    except ValueError as exc:
        raise gr.Error(
            "The sentiment API returned invalid JSON."
        ) from exc

    scores = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(scores, dict) or not scores:
        raise gr.Error(
            "The sentiment API returned no class scores."
        )

    return {str(label): float(score) for label, score in scores.items()}


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

if __name__ == "__main__":
    demo.launch(theme=theme)
