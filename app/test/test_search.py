# app/test/test_search.py
import os
import pytest
import logging
import numpy as np
from pathlib import Path
import tempfile
import asyncio
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.core.config.config import Config
from app.core.faiss_manager import FAISSVectorStore
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.embedding_repository import EmbeddingRepository
from app.ingestion.chunk_repository import ChunkRepository
from app.query.query_processor import QueryProcessor
from app.query.retriever import Retriever
from app.query.context_builder import ContextBuilder

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
def test_config():
    """Fixture para crear una configuración de prueba."""
    return Config()

@pytest.fixture
def embedding_service(test_config):
    """Fixture para crear un servicio de embeddings."""
    return EmbeddingService(test_config)

@pytest.fixture
def test_indices_dir():
    """Fixture para crear un directorio temporal para índices FAISS."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def isolated_faiss_store(test_config, test_indices_dir):
    """Fixture para crear un almacén FAISS aislado."""
    config = test_config
    config.INDICES_DIR = Path(test_indices_dir)
    return FAISSVectorStore(config)

@pytest.fixture
def faiss_store(test_config):
    """Fixture para crear un almacén FAISS."""
    return FAISSVectorStore(test_config)

@pytest.fixture
def embedding_repository(test_config):
    """Fixture para crear un repositorio de embeddings."""
    return EmbeddingRepository(test_config)

@pytest.fixture
def chunk_repository(test_config):
    """Fixture para crear un repositorio de chunks."""
    return ChunkRepository(test_config)

@pytest.fixture
def query_processor(test_config, embedding_service):
    """Fixture para crear un procesador de consultas."""
    return QueryProcessor(test_config, embedding_service)

@pytest.fixture
def retriever(faiss_store, embedding_repository, test_config):
    """Fixture para crear un recuperador."""
    return Retriever(faiss_store, embedding_repository, test_config)

@pytest.fixture
def context_builder(test_config, chunk_repository):
    """Fixture para crear un constructor de contexto."""
    return ContextBuilder(test_config, chunk_repository)

def test_query_processor_normalization(query_processor):
    """Prueba la normalización de consultas."""
    # Probar normalización básica
    original = " ¿Qué   lugares turísticos hay en Curazao? "
    normalized = query_processor.normalize_query(original)
    assert normalized
    assert len(normalized) < len(original)
    assert "?" in normalized
    
    # Probar con caracteres especiales
    original = "¿Dónde encontrar playas @Curazao! #turismo"
    normalized = query_processor.normalize_query(original)
    assert normalized
    assert "@" not in normalized
    assert "#" not in normalized

def test_query_processor_detect_language(query_processor):
    """Prueba la detección de idioma."""
    # Español
    query_es = "¿Cuáles son las mejores playas de Curazao?"
    lang = query_processor._detect_language(query_es)
    assert lang == "es"
    
    # Inglés
    query_en = "What are the best beaches in Curacao?"
    lang = query_processor._detect_language(query_en)
    assert lang == "en"

def test_query_processor_detect_type(query_processor):
    """Prueba la detección de tipo de consulta."""
    # Pregunta
    query_question = "¿Dónde está el mejor restaurante?"
    q_type = query_processor._detect_query_type(query_question)
    assert q_type == "question"
    
    # Búsqueda
    query_search = "buscar hoteles en willemstad"
    q_type = query_processor._detect_query_type(query_search)
    assert q_type == "search"
    
    # General
    query_general = "Información sobre Curazao"
    q_type = query_processor._detect_query_type(query_general)
    assert q_type == "general"

def test_query_processor_preprocess(query_processor):
    """Prueba el preprocesamiento completo de consultas."""
    query = "¿Qué lugares turísticos hay en Curazao para visitar?"
    result = query_processor.preprocess_query(query)
    
    assert result["original_query"] == query
    assert "processed_query" in result
    assert "language" in result
    assert "query_type" in result
    assert "token_count" in result
    assert result["query_type"] == "question"

def test_generate_embedding(embedding_service, query_processor):
    """Prueba la generación de embeddings para consultas."""
    query = "Playas de Curazao"
    embedding = query_processor.generate_embedding(query)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 384  # Dimensión de miniLM
    
    # Verificar que los valores son normales
    assert np.all(np.isfinite(embedding))
    assert np.any(embedding != 0)  # No todos son cero

def test_faiss_search_basic(isolated_faiss_store):
    """Prueba búsqueda básica con FAISS."""
    # Verificamos que empezamos con un índice vacío
    initial_info = isolated_faiss_store.get_index_info()
    assert initial_info['total_vectors'] == 0, "El índice FAISS no está vacío inicialmente"
    
    # Añadir algunos vectores de prueba
    test_vectors = np.random.rand(5, 384).astype(np.float32)
    ids = isolated_faiss_store.add_vectors(test_vectors)
    
    # Crear vector de consulta
    query_vector = np.random.rand(1, 384).astype(np.float32)
    
    # Realizar búsqueda
    distances, indices = isolated_faiss_store.search(query_vector, k=3)
    
    assert len(indices[0]) == 3
    assert len(distances[0]) == 3
    assert np.all(indices[0] >= 0)
    assert np.all(indices[0] < 5)

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_retriever_faiss_search(retriever):
    """Prueba la búsqueda con FAISS desde el Retriever."""
    # Generar vector de prueba
    query_vector = np.random.rand(384).astype(np.float32)
    
    # Realizar búsqueda
    results = await retriever.search_faiss(query_vector, k=5)
    
    assert "indices" in results
    assert "distances" in results
    assert "similarities" in results
    assert "search_time" in results
    assert results["mode"] == "faiss"

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_context_building(context_builder):
    """Prueba la construcción de contexto a partir de chunk_ids."""
    # Generar chunk_ids simulados (en un sistema real se obtendrían del Retriever)
    chunk_ids = ["123e4567-e89b-12d3-a456-426614174000", 
                "123e4567-e89b-12d3-a456-426614174001"]
    similarities = [0.85, 0.72]
    
    # Construir contexto
    context = await context_builder.build_context_by_relevance(chunk_ids, similarities)
    
    # Verificar estructura
    assert "context" in context
    assert "chunks" in context
    assert "strategy" in context
    assert context["strategy"] == "relevance"

@pytest.mark.skip("Esta prueba verifica la base de datos")
def test_postgres_search_tables():
    """Verifica que las tablas necesarias para la búsqueda existen."""
    # Verificar tabla de chunks
    chunks_table = execute_sql("SELECT column_names FROM information_schema.columns WHERE table_name = 'chunks'")
    assert chunks_table is not None
    
    # Verificar extensión pgvector
    vector_extension = execute_sql("SELECT * FROM pg_extension WHERE extname = 'vector'")
    assert vector_extension is not None
    assert "vector" in vector_extension

if __name__ == "__main__":
    pytest.main(["-v", "test_search.py"])