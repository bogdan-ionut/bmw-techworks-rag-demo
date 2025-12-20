FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Removed build-essential and curl to reduce image size.
# Most python packages have wheels for 3.11-slim (Debian Bookworm).
# If a package needs compilation, we can use a multi-stage build, but for now this is leaner.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
