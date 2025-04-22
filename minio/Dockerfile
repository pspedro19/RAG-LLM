# Use the official FastAPI image as a base
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

# Install necessary packages before installing Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your application files
COPY ./app /app

# Expose the port the app runs on
EXPOSE 8800

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800"]
