# Use an official Python slim image as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /mlflow

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    postgresql \
    postgresql-contrib \
    postgresql-client \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file and the database initialization script into the container
COPY requirements.txt .
COPY init_db.sh .

# Fix line endings and make the script executable
RUN sed -i 's/\r$//' init_db.sh && \
    chmod +x init_db.sh

# Install Python dependencies along with MLflow and psycopg2 for PostgreSQL interaction
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlflow[extras,sqlalchemy,google] psycopg2-binary boto3 cryptography pymysql

# Expose the port MLflow will use
EXPOSE 5000

# Define environment variables for PostgreSQL connection
ENV POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=postgres_password \
    POSTGRES_DB=mlflow_db \
    POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_APP_USER=airflow \
    POSTGRES_APP_PASSWORD=airflow

# Ensure the database is initialized before starting MLflow
ENTRYPOINT ["./init_db.sh"]

# Command to run MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql+psycopg2://airflow:airflow@postgres:5432/mlflow_db", \
     "--default-artifact-root", "s3://minio:9000/mlflow", \
     "--host", "0.0.0.0", \
     "--port", "5000"]