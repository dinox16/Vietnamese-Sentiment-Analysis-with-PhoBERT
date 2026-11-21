---
title: Student Feedback Sentiment Classifier
colorFrom: blue
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# Student Feedback Sentiment Classifier

This project uses **PhoBERT-base**, fine-tuned on the **UIT-VSFC (Vietnamese Students' Feedback Corpus)**, to classify student comments as **negative, neutral, or positive**. Enter a comment in Vietnamese to see the model's predicted sentiment and confidence scores.

- **Live demo:** `https://huggingface.co/spaces/YOUR_HF_USERNAME/phobert-uit-vsfc-demo` (replace `YOUR_HF_USERNAME`)
- **Fine-tuned model:** `https://huggingface.co/YOUR_HF_USERNAME/phobert-uit-vsfc` (replace `YOUR_HF_USERNAME`)
- **Training notebook:** [notebooks/phobert_uit_vsfc_train.ipynb](notebooks/phobert_uit_vsfc_train.ipynb)

## Test results

The best checkpoint was selected using validation Macro-F1. The figures below are from the held-out test set, as recorded in `result/DTU-CS/HGK/results/test_metrics.json`.

| Model | Test accuracy | Test Macro-F1 |
|---|---:|---:|
| PhoBERT-base | 94.19% | 0.8380 |

Training curves and the test confusion matrix are available in [`result/DTU-CS/HGK/results`](result/DTU-CS/HGK/results/).

## How it works

The app normalizes Vietnamese text and segments words with `underthesea`, matching the preprocessing used during training. PhoBERT then returns a score for each of the three sentiment labels. The dataset is `uitnlp/vietnamese_students_feedback`; neutral examples are relatively rare, so Macro-F1 is useful alongside accuracy when assessing performance across classes.

The saved training configuration is in [`result/DTU-CS/HGK/results/config.json`](result/DTU-CS/HGK/results/config.json). It records a maximum sequence length of 128, five epochs, a learning rate of `2e-5`, and a warmup ratio of `0.1`.

## Project structure

```text
app.py                         Gradio web demo
vsfc/preprocessing.py          Text normalization and word segmentation
vsfc/predictor.py              Tokenization, model loading, and prediction
scripts/predict.py             Command-line prediction
notebooks/                     Training notebook
result/DTU-CS/HGK/best_model/  Fine-tuned model and tokenizer files
result/DTU-CS/HGK/results/     Metrics, configuration, plots, and error examples
tests/                         Preprocessing tests
```

## Run locally

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/vsfc-sentiment.git
cd vsfc-sentiment
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The app calls the model through the Hugging Face Inference API. Copy `.env.example` to `.env`, then add your Hugging Face access token and model ID:

```bash
cp .env.example .env
```

Edit `.env` and replace `HF_TOKEN` with your Hugging Face token. Set `HF_MODEL_ID` to the repository containing your fine-tuned text-classification model. Keep `.env` private; it is excluded from Git. Install the dependencies with `pip install -r requirements.txt`, then start the app:

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser. The Hugging Face Inference API must be available for the selected model; usage may depend on your Hugging Face account and plan.

For a command-line prediction:

```bash
python scripts/predict.py --model result/DTU-CS/HGK/best_model "Thầy dạy rất hay"
```

## Train again

Open the notebook in `notebooks/` in an environment with the required dependencies and a GPU if available. Its recorded configuration and generated artifacts are under `result/DTU-CS/HGK/`.

## References

- Nguyen et al. (2018). *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis.* KSE 2018.
- Nguyen and Nguyen (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP 2020.
