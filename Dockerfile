FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt "uvicorn[standard]" python-dotenv

COPY . .
RUN mkdir -p /app/results && useradd -m rescuesim && chown -R rescuesim /app
USER rescuesim

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"

# Interactive command center (Nebius Serverless Endpoint).
CMD ["python", "server.py"]

# Nebius Serverless Job: override the command, e.g.
#   python -m nebius_jobs.batch_policy_eval --episodes 100 --workers 8 --output /results/serverless_eval.json