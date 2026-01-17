# Use Python 3.9 slim image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Run with gunicorn
# --workers 1: Use single worker to keep models in memory
# --threads 4: Handle multiple requests with threads
# --timeout 300: Allow 5 minutes for requests (model loading + prediction)
CMD exec gunicorn --bind :8080 --workers 1 --threads 4 --timeout 300 app:app