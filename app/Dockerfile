FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \ 
    && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /app/chat-Interface /app/images

# Copiar requirements primero (para mejor uso de caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar modelos de spaCy
RUN python -m pip install --no-cache-dir spacy
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_md
RUN python -m spacy validate

# Instalar uvicorn
RUN pip install uvicorn

# Copiar el resto de la aplicación
COPY . .

# Configurar PATHS
ENV PYTHONPATH=/app
ENV MEDIA_DIR=/app/images
ENV DB_PATH=/app/chat-Interface/db.sqlite3

# Puerto y comando
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]