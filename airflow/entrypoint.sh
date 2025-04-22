#!/bin/bash

# Espera hasta que PostgreSQL esté disponible
until pg_isready -h postgres -p 5432 -U "airflow"; do
    echo "Waiting for PostgreSQL to start..."
    sleep 2
done

# Verifica si la base de datos necesita ser inicializada
if [ -z "$(psql -Atqc "\\list airflow")" ]; then
    echo "Database not found, initializing..."
    airflow db init
fi

# Inicia el servidor web de Airflow
# Run in compose
# exec airflow webserver
