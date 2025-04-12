FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Default Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the app
CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true 