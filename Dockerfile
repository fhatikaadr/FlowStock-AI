FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    APP_DIR="AI Recommendation"

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r "${APP_DIR}/requirements.txt"

CMD ["sh", "-c", "cd \"$APP_DIR\" && uvicorn main:app --host 0.0.0.0 --port ${PORT}"]