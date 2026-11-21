"""Gradio demo for Vietnamese student feedback sentiment classification."""

import os

import gradio as gr

from vsfc.predictor import SentimentPredictor

# Set MODEL_ID to your Hugging Face model repository or local model directory.
# The bundled checkpoint can be selected with:
# MODEL_ID=result/DTU-CS/HGK/best_model python app.py
MODEL_ID = os.getenv("MODEL_ID", "Dinox16/phobert-sentiment")
LABEL_EN = {"Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}

predictor = SentimentPredictor(MODEL_ID)


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {}
    result = predictor.predict(text)[0]
    return {LABEL_EN.get(label, label): score for label, score in result["probabilities"].items()}


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


if __name__ == "__main__":
    demo.launch()
