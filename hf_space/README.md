---
title: PhoBERT Sentiment API
emoji: 💬
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# PhoBERT Sentiment API

Inference API for Vietnamese student feedback. Send a `POST` request to `/predict` with a JSON body such as `{"text":"Thầy giảng bài rất dễ hiểu."}`. The response contains the predicted label and a score for each class.

The service loads `Dinox16/phobert-sentiment` by default. Set `HF_MODEL_ID` in the Space variables to use a different model. If the model repository is private or gated, add `HF_TOKEN` as a Space secret.
