# StockWise AI dashboard image (Level 4).
# Build from the project root:  docker build -f deploy/Dockerfile -t stockwise-dashboard .
# Run:                          docker run -d -p 8501:8501 stockwise-dashboard
FROM python:3.12-slim

WORKDIR /app

# 1. Libraries first (this layer is cached when only the code changes)
COPY deploy/requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# 2. The dashboard code and its result files (dashboard.py reads ROOT / "reports")
COPY app/ app/
COPY deploy/data/ reports/

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
