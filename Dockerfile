FROM python:3.11-slim

WORKDIR /app

# System dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data charts

# Default: run the interactive Telegram bot
CMD ["python", "scripts/run_bot.py"]
