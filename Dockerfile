# Use official Python runtime as base image (updated to 3.11 for better compatibility)
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Set environment variables for secure application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/models_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Install system dependencies including SQLite for user authentication
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories for user data and model cache
RUN mkdir -p /app/data /app/models_cache /app/user_data

# Set proper permissions for SQLite database
RUN chmod 755 /app && chmod 644 /app/*.py

# Expose Streamlit port
EXPOSE 8501

# Health check for secure application
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Create startup script for proper initialization
RUN echo '#!/bin/bash\n\
# Initialize SQLite database if it doesn\'t exist\n\
if [ ! -f /app/user_data/users.db ]; then\n\
    echo "Initializing user database..."\n\
    python -c "from user_auth import UserAuthSystem; UserAuthSystem()"\n\
fi\n\
# Start Streamlit application\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh \
    && chmod +x /app/start.sh

# Run secure E-Commerce Review Analyzer
ENTRYPOINT ["/app/start.sh"]
