# app/test/test_sync.py
import os
import pytest
import asyncio
import logging
import numpy as np
from pathlib import Path
import uuid
import tempfile
import shutil
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.core.config.config import Config
from app.ingestion.chunk_repository import ChunkRepository
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.embedding_repository import EmbeddingRepository
from app.core.faiss_manager import FAISSVectorStore
from app.core.sync.service import SyncService
from app.core.sync.monitoring import SyncMonitor

def execute_sql(sql_command):
    """Ejecuta un comando SQL en la base de datos PostgreSQL usando Docker."""
    try:
        cmd = [
            "docker", "exec", "-i", "postgres_pgvector",
            "psql", "-U", "myuser", "-d", "mydatabase", "-c", sql_command
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing SQL: {e}")
        logger.error(f"SQL stderr: {e.stderr}")
        return None

@pytest.fixture
def test_indices_dir():
    """Fixture para crear un directorio temporal para índices FAISS."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(test_indices_dir):
    """Fixture para crear una configuración de prueba."""
    config = Config()
    config.INDICES_DIR = Path(test_indices_dir)
    return config

@pytest.fixture
def faiss_store(test_config):
    """Fixture para crear un almacén FAISS."""
    return FAISSVectorStore(test_config)

@pytest.fixture
def embedding_service(test_config):
    """Fixture para crear un servicio de embeddings."""
    return EmbeddingService(test_config)

@pytest.fixture
def sync_service(test_config, faiss_store):
    """Fixture para crear un servicio de sincronización."""
    return SyncService(test_config, faiss_store)

@pytest.fixture
def sync_monitor(sync_service, test_config):
    """Fixture para crear un monitor de sincronización."""
    return SyncMonitor(sync_service, test_config)

def test_faiss_store_operations(faiss_store):
    """Prueba operaciones básicas del almacén FAISS."""
    # Crear vectores de prueba
    vectors = np.random.rand(3, 384).astype(np.float32)
    
    # Añadir vectores
    ids = faiss_store.add_vectors(vectors)
    
    # Verificar que se añadieron correctamente
    assert len(ids) == 3
    
    # Realizar una búsqueda
    query = np.random.rand(1, 384).astype(np.float32)
    distances, indices = faiss_store.search(query, k=2)
    
    # Verificar resultados
    assert indices.shape == (1, 2)
    assert distances.shape == (1, 2)
    
    # Obtener información del índice
    info = faiss_store.get_index_info()
    assert info['total_vectors'] == 3
    assert info['dimension'] == 384

@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
def test_sync_service_functions(sync_service):
    """Prueba las funciones del servicio de sincronización."""
    # Verificar que el servicio está inicializado correctamente
    assert sync_service is not None
    assert sync_service.faiss_store is not None
    assert sync_service.is_syncing == False
    
    # Probar parsing de embeddings
    test_embedding = [0.1] * 384
    parsed = sync_service._parse_embedding(test_embedding)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape == (1, 384)
    
    # Verificar info del sistema
    status = sync_service.get_system_status()
    assert 'is_syncing' in status
    assert 'stats' in status

def test_sync_monitor_initialization(sync_monitor):
    """Prueba la inicialización del monitor de sincronización."""
    assert sync_monitor is not None
    assert sync_monitor.sync_service is not None
    assert sync_monitor.is_monitoring == False
    
    metrics = sync_monitor.get_metrics()
    assert metrics['total_syncs'] == 0
    assert metrics['total_checks'] == 0
    assert metrics['is_monitoring'] == False

@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
def test_postgres_vector_tables():
    """Verifica que las tablas necesarias para pgvector existen."""
    # Verificar la existencia de tablas
    tables = execute_sql("\\dt")
    assert tables is not None
    
    # Verificar que la extensión pgvector está instalada
    extensions = execute_sql("SELECT * FROM pg_extension WHERE extname = 'vector'")
    assert extensions is not None
    assert "vector" in extensions
    
    # Verificar estructura de la tabla embeddings
    embedding_cols = execute_sql(
        "SELECT column_name, data_type FROM information_schema.columns " +
        "WHERE table_name = 'embeddings' ORDER BY ordinal_position"
    )
    assert embedding_cols is not None
    
    # Debería ver columnas como 'embedding_id', 'chunk_id', 'embedding', 'model_name', 'faiss_index_id'
    assert "embedding_id" in embedding_cols
    assert "chunk_id" in embedding_cols

if __name__ == "__main__":
    pytest.main(["-v", "test_sync.py"])