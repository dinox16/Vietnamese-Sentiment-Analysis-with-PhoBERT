FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY hf_space/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY vsfc ./vsfc
COPY hf_space/app.py ./app.py

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
