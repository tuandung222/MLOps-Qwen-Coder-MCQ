# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Set environment variables
ENV MODEL_PATH="tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
ENV DEVICE="cuda:0"
ENV MAX_LENGTH="2048"
ENV JAEGER_HOST="jaeger"
ENV JAEGER_PORT="6831"

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]