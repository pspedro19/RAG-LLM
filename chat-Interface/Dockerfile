FROM python:3.9

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install django --upgrade
RUN pip install whitenoise

# Crear directorios necesarios
RUN mkdir -p /app/staticfiles
RUN mkdir -p /app/static
RUN mkdir -p /app/static/admin/css
RUN mkdir -p /app/static/admin/js
RUN mkdir -p /app/static/admin/img
RUN mkdir -p /app/property_images

# Copiar el proyecto
COPY . .

# Establecer permisos
RUN chmod -R 755 /app
RUN chmod -R 755 /app/staticfiles
RUN chmod -R 755 /app/static

# Recolectar archivos estáticos con verbose
RUN python manage.py collectstatic --noinput --clear -v 2

EXPOSE 8000

CMD ["gunicorn", "chatbot.wsgi:application", "--bind", "0.0.0.0:8000"]