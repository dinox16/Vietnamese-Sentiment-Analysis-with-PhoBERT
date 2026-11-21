---
title: Student Feedback Sentiment Classifier
colorFrom: blue
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# Student Feedback Sentiment Classifier

This project uses **PhoBERT-base**, fine-tuned on the **UIT-VSFC (Vietnamese Students' Feedback Corpus)**, to classify student comments as **negative, neutral, or positive**. The Gradio demo and JSON inference API are served by the same `app.py`.

- **Live demo:** [Vercel website](https://vietnamese-sentiment-analysis-with-one.vercel.app/)
- **Hugging Face Space:** [Dinox16/pho-bert-DTU](https://huggingface.co/spaces/Dinox16/pho-bert-DTU)
- **Fine-tuned model:** [Dinox16/phobert-sentiment](https://huggingface.co/Dinox16/phobert-sentiment)
- **Training notebook:** [notebooks/dtu-sda-da-cs467a-hogiakhanh-havanquoctien (1).ipynb](notebooks/dtu-sda-da-cs467a-hogiakhanh-havanquoctien%20(1).ipynb)

## Test results

The best checkpoint was selected using validation Macro-F1. The held-out test results are recorded in `result/DTU-CS/HGK/results/test_metrics.json`.

| Model | Test accuracy | Test Macro-F1 |
|---|---:|---:|
| PhoBERT-base | 94.19% | 0.8380 |

Training curves and the test confusion matrix are in [`result/DTU-CS/HGK/results`](result/DTU-CS/HGK/results/).

## How it works

The app normalizes Vietnamese text and segments words with `underthesea`, matching the training preprocessing. PhoBERT returns scores for the three sentiment labels. The dataset is `uitnlp/vietnamese_students_feedback`; neutral examples are relatively rare, so Macro-F1 helps assess performance across all classes.

The saved training configuration is in [`result/DTU-CS/HGK/results/config.json`](result/DTU-CS/HGK/results/config.json). It records a maximum sequence length of 128, five epochs, a learning rate of `2e-5`, and a warmup ratio of `0.1`.

## Project structure

```text
app.py                         Gradio interface and FastAPI inference endpoints
vsfc/preprocessing.py          Text normalization and word segmentation
vsfc/predictor.py              Local command-line predictor
scripts/predict.py             Command-line prediction
notebooks/                     Training notebook
result/DTU-CS/HGK/results/     Metrics, configuration, plots, and error examples
tests/                         Preprocessing tests
```

## Run locally

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `HF_MODEL_ID` to the model repository. If it is private or gated, set `HF_TOKEN` as well. Keep `.env` private; it is excluded from Git. Then run:

```bash
python app.py
```

Open `http://127.0.0.1:7860` for the Gradio interface. The same app exposes `GET /health` and `POST /predict`. Send a JSON body such as `{"text":"Thầy giảng bài rất dễ hiểu."}` to `/predict`; the response includes `text`, `prediction`, and `scores`.

## Deploy to Hugging Face Spaces

Deploy this repository as a Gradio Space using `app.py` as the entry point. Set `HF_MODEL_ID` in Space variables, and add `HF_TOKEN` as a Space secret only if the model is private or gated. The Space serves the demo and inference API together. Configure the Vercel server to call `https://<your-space-subdomain>.hf.space/predict`. Keep any private Space access token in Vercel server environment variables, never in browser code.

## Train again

Open the notebook in `notebooks/` in an environment with the required dependencies and a GPU if available. Training artifacts are stored under `result/DTU-CS/HGK/`.

## References

- Nguyen et al. (2018). *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis.* KSE 2018.
- Nguyen and Nguyen (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP 2020.
