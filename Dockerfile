FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/default.yaml"]
