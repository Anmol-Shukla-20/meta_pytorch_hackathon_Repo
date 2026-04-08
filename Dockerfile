FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Run inference
CMD ["python", "inference.py"]