FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create a health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8000/api/health || exit 1' > /healthcheck.sh \
    && chmod +x /healthcheck.sh

EXPOSE 8000 8501

# Default command
CMD ["python", "backend/app.py"]