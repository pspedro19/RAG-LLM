# app/test/test_config.py
import os
import pytest
import asyncio
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.core.config.config import Config

@pytest.fixture
def test_config():
    """Fixture para crear una configuración de prueba."""
    # Establecer variables de entorno para testing
    os.environ["DB_HOST"] = "postgres_pgvector"  # Nombre del contenedor Docker
    os.environ["DB_PORT"] = "5432"
    os.environ["DB_NAME"] = "mydatabase"
    os.environ["DB_USER"] = "myuser"
    os.environ["DB_PASSWORD"] = "mypassword"
    
    config = Config()
    return config

def test_config_loads_environment_variables(test_config):
    """Verifica que la configuración cargue correctamente las variables de entorno."""
    assert test_config.DB_HOST == "postgres_pgvector"
    assert test_config.DB_PORT == 5432  # Debería convertirse a entero
    assert test_config.DB_NAME == "mydatabase"
    assert test_config.DB_USER == "myuser"
    assert test_config.DB_PASSWORD == "mypassword"

def test_config_default_values(test_config):
    """Verifica que los valores por defecto se establezcan correctamente."""
    assert hasattr(test_config, "CHUNK_SIZE")
    assert hasattr(test_config, "CHUNK_OVERLAP")
    assert hasattr(test_config, "INDICES_DIR")

@pytest.mark.asyncio
async def test_database_connection():
    """Prueba la conexión a la base de datos usando la configuración."""
    # Usar "docker exec" para verificar las tablas directamente en el contenedor
    import subprocess
    import json
    
    config = Config()
    logger.info(f"Intentando conectar a PostgreSQL en: {config.DB_HOST}:{config.DB_PORT}")
    
    try:
        # Verificar que el contenedor esté en ejecución
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=postgres_pgvector", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        if "postgres_pgvector" not in result.stdout:
            pytest.skip("El contenedor postgres_pgvector no está en ejecución")
            return
        
        # Listar tablas usando docker exec
        cmd = [
            "docker", "exec", "postgres_pgvector", 
            "psql", "-U", config.DB_USER, "-d", config.DB_NAME,
            "-c", "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Error al ejecutar comando en PostgreSQL: {result.stderr}")
        
        logger.info("Tablas encontradas:")
        for line in result.stdout.strip().split('\n'):
            if "documents" in line or "chunks" in line or "embeddings" in line:
                logger.info(f"  - {line.strip()}")
        
        # Verificar extensión pgvector
        cmd = [
            "docker", "exec", "postgres_pgvector", 
            "psql", "-U", config.DB_USER, "-d", config.DB_NAME,
            "-c", "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Error al verificar extensión pgvector: {result.stderr}")
        
        # Verificar que haya al menos 1 extensión llamada 'vector'
        if "1" in result.stdout:
            logger.info("Extensión pgvector verificada correctamente")
        else:
            raise Exception("La extensión pgvector no está instalada")
        
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos: {e}")
        pytest.fail(f"Error de conexión: {e}")

if __name__ == "__main__":
    asyncio.run(test_database_connection())