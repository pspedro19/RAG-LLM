CREATE USER airflow WITH PASSWORD 'airflow';
ALTER USER airflow WITH SUPERUSER;
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
