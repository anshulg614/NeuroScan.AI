FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Remove explicit port exposure since Render will assign its own
# EXPOSE 8501

# Remove healthcheck since it might conflict with Render's own healthcheck
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the app
CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 