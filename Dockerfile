FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Removed apt-get update to avoid Docker Buildkit disk IO error
# Relying on binary wheels for dependencies

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Explicitly upgrade datasets to avoid the Python 3.11 dataclass bug
RUN pip install --no-cache-dir --upgrade datasets==5.0.0

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application with 4 uvicorn workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
