FROM python:3.11-slim

# CBC solver ships inside the pulp wheel; coinor-cbc is a safety net.
RUN apt-get update \
 && apt-get install -y --no-install-recommends coinor-cbc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# GEMINI_API_KEY is injected at runtime (-e or --env-file), never baked in.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
