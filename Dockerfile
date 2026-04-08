FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Create cache directory
RUN mkdir -p cache

# HF Spaces requires port 7860
EXPOSE 7860

# Run on port 7860 — this is mandatory for HF Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]