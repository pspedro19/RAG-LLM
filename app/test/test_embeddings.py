# app/test/test_embeddings.py
import os
import pytest
import asyncio
import logging
import numpy as np
from pathlib import Path
import uuid
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.core.config.config import Config
from app.embeddings.embedding_service import EmbeddingService, LocalModelEmbedder
from app.embeddings.embedding_repository import EmbeddingRepository
from app.ingestion.chunk_repository import ChunkRepository

@pytest.fixture
def test_config():
    """Fixture para crear una configuración de prueba."""
    config = Config()
    return config

@pytest.fixture
def embedding_service(test_config):
    """Fixture para crear un servicio de embeddings."""
    return EmbeddingService(test_config)

@pytest.fixture
async def embedding_repository(test_config):
    """Fixture para crear un repositorio de embeddings."""
    repo = EmbeddingRepository(test_config)
    yield repo
    await repo.close()

@pytest.fixture
async def chunk_repository(test_config):
    """Fixture para crear un repositorio de chunks."""
    repo = ChunkRepository(test_config)
    yield repo
    await repo.close()

def execute_sql(config, sql_command):
    """Ejecuta un comando SQL en el contenedor de PostgreSQL."""
    cmd = [
        "docker", "exec", "postgres_pgvector", 
        "psql", "-U", config.DB_USER, "-d", config.DB_NAME,
        "-c", sql_command
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error ejecutando SQL: {result.stderr}")
    return result.stdout

@pytest.fixture
async def test_document_with_chunks(test_config):
    """Fixture para crear un documento de prueba con chunks."""
    # Verificar que el contenedor está ejecutándose
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=postgres_pgvector", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if "postgres_pgvector" not in result.stdout:
        pytest.skip("El contenedor postgres_pgvector no está en ejecución")
    
    # Crear documento directamente con SQL
    doc_id = str(uuid.uuid4())
    execute_sql(test_config, f"""
        INSERT INTO documents (doc_id, title, metadata)
        VALUES ('{doc_id}', 'Documento de prueba para embeddings', '{{"category": "test_embeddings"}}')
    """)
    
    # Crear chunks
    chunk_ids = []
    chunks_data = [
        (0, "Este es el primer chunk de prueba para embeddings.", '{"position": "first"}'),
        (1, "Este es el segundo chunk con contenido diferente.", '{"position": "second"}'),
        (2, "Este es el tercer y último chunk de este documento.", '{"position": "last"}')
    ]
    
    for chunk_number, content, metadata in chunks_data:
        chunk_id = str(uuid.uuid4())
        execute_sql(test_config, f"""
            INSERT INTO chunks (chunk_id, doc_id, content, chunk_number, metadata)
            VALUES ('{chunk_id}', '{doc_id}', '{content}', {chunk_number}, '{metadata}')
        """)
        chunk_ids.append(chunk_id)
    
    yield {'doc_id': doc_id, 'chunk_ids': chunk_ids}
    
    # Limpiar después de las pruebas
    execute_sql(test_config, f"DELETE FROM documents WHERE doc_id = '{doc_id}'")

def test_local_embedder_initialization():
    """Prueba la inicialización de un embedder local."""
    embedder = LocalModelEmbedder()
    
    # Verificar propiedades
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.dimension == 384
    assert callable(embedder.embed_texts)

def test_embedding_generation(embedding_service):
    """Prueba la generación de embeddings para textos."""
    texts = [
        "Este es un texto de prueba.",
        "Este es otro texto diferente para probar embeddings."
    ]
    
    # Generar embeddings
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Verificar resultados
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0  # Dimensión del embedding
    
    # Verificar que los embeddings son diferentes
    assert not np.array_equal(embeddings[0], embeddings[1])

def test_embedding_service_models(embedding_service):
    """Prueba la información de modelos disponibles."""
    models_info = embedding_service.get_models_info()
    
    # Verificar que hay al menos un modelo
    assert len(models_info) > 0
    
    # Verificar estructura de la información
    for model in models_info:
        assert 'key' in model
        assert 'name' in model
        assert 'dimension' in model
        assert 'type' in model

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_embedding_storage_and_retrieval(embedding_service, embedding_repository, test_document_with_chunks):
    """Prueba el almacenamiento y recuperación de embeddings."""
    chunk_ids = test_document_with_chunks['chunk_ids']
    
    # Verificar que tenemos chunks para trabajar
    assert len(chunk_ids) > 0
    
    # Generar embeddings para los chunks
    texts = ["Texto para el embedding del primer chunk"]
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Almacenar el embedding para el primer chunk
    embedding_id = await embedding_repository.store_embedding(
        chunk_id=chunk_ids[0],
        embedding=embeddings[0],
        model_name='miniLM'
    )
    
    # Verificar que se creó el embedding
    assert embedding_id is not None
    
    # Recuperar el embedding
    stored_embedding = await embedding_repository.get_embedding(embedding_id)
    
    # Verificar resultados
    assert stored_embedding is not None
    assert stored_embedding['chunk_id'] == chunk_ids[0]
    assert stored_embedding['model_name'] == 'miniLM'
    assert isinstance(stored_embedding['embedding'], np.ndarray)
    assert len(stored_embedding['embedding']) == len(embeddings[0])

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_vector_search(embedding_service, embedding_repository, test_document_with_chunks):
    """Prueba la búsqueda vectorial con pgvector."""
    try:
        # Verificar si pgvector está habilitado para búsqueda
        pool = await embedding_repository._get_pool()
        async with pool.acquire() as conn:
            has_vector_extension = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            )
        
        if not has_vector_extension:
            pytest.skip("La extensión pgvector no está disponible")
    except Exception as e:
        pytest.skip(f"Error verificando pgvector: {e}")
    
    chunk_ids = test_document_with_chunks['chunk_ids']
    
    # Generar embeddings para todos los chunks
    texts = [
        "Este es el primer chunk para búsqueda.",
        "Este es el segundo chunk con contenido distinto.",
        "Este es el tercer chunk para probar búsqueda."
    ]
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Almacenar embeddings
    embedding_ids = []
    for i, chunk_id in enumerate(chunk_ids):
        if i < len(embeddings):
            emb_id = await embedding_repository.store_embedding(
                chunk_id=chunk_id,
                embedding=embeddings[i],
                model_name='miniLM'
            )
            embedding_ids.append(emb_id)
    
    # Verificar que se crearon los embeddings
    assert len(embedding_ids) > 0
    
    # Crear un vector de consulta similar al primer texto
    query_text = "Buscar el primer chunk"
    query_vector = embedding_service.generate_embeddings([query_text])[0]
    
    # Realizar búsqueda
    search_results = await embedding_repository.search_similar(
        query_vector=query_vector,
        k=3,
        distance_threshold=0.5
    )
    
    # Verificar resultados
    assert len(search_results) > 0
    
    # El primer resultado debería ser similar al primer chunk
    if search_results:
        logger.info(f"Similitud más alta: {search_results[0]['similarity']}")
        logger.info(f"Contenido encontrado: {search_results[0]['content']}")

if __name__ == "__main__":
    pytest.main(["-v", "test_embeddings.py"])