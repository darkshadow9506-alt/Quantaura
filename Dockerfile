# QuantAura Telegram signal bot
FROM python:3.11-slim

# faster, quieter Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# persist the cache and the journal database on a mounted volume
ENV QUANTAURA_DB=/app/state/quantaura_state.db
RUN mkdir -p /app/state /app/data_cache

# sanity-check the build (offline, no network needed)
RUN python -m quantaura selftest

CMD ["python", "-m", "quantaura", "bot"]
