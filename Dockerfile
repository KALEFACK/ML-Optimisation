FROM python:3.11-slim

LABEL maintainer="Groupe G11"
LABEL description="Dashboard P03 - Transfert Cross-Lingue DistilBERT vs CamemBERT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8504 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY runs/       ./runs/
COPY figures/    ./figures/

RUN mkdir -p /app/.streamlit
COPY .streamlit/ ./.streamlit/

EXPOSE 8504

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8504/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8504", "--server.address=0.0.0.0"]
