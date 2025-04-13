FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to suppress warnings
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove explicit port exposure since Render will assign its own
# EXPOSE 8501

# Remove healthcheck since it might conflict with Render's own healthcheck
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the app
CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 