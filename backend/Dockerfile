FROM python:3.10-slim

WORKDIR /app

# Install dependencies for statsmodels, prophet and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
