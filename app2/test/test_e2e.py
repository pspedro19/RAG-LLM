# app2/test/test_e2e.py
import os
import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app2.core.config.config import Config
from app2.ingestion.document_reader import DocumentReader
from app2.ingestion.document_processor import DocumentProcessor
from app2.ingestion.chunk_repository import ChunkRepository
from app2.embeddings.embedding_service import EmbeddingService
from app2.embeddings.embedding_repository import EmbeddingRepository
from app2.core.faiss_manager import FAISSVectorStore
from app2.core.sync.service import SyncService
from app2.query.query_processor import QueryProcessor
from app2.query.retriever import Retriever
from app2.query.context_builder import ContextBuilder

@pytest.fixture
def test_indices_dir():
    """Fixture para crear un directorio temporal para índices FAISS."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_docs_dir():
    """Fixture para crear un directorio temporal para documentos de prueba."""
    temp_dir = tempfile.mkdtemp()
    
    # Crear algunos archivos de texto para las pruebas
    files = [
        ("historia_paris.txt", """
        París es la capital de Francia y una de las ciudades más importantes de Europa. 
        Su historia se remonta a más de 2000 años, cuando era un asentamiento galo conocido como Lutecia.
        Tras la conquista romana, la ciudad comenzó a expandirse en la Île de la Cité.
        Durante la Edad Media, París se convirtió en un importante centro cultural y educativo con la fundación de la Universidad de París.
        """),
        ("torre_eiffel.txt", """
        La Torre Eiffel es el símbolo más reconocible de París. Fue construida entre 1887 y 1889 por Gustave Eiffel.
        Originalmente criticada por muchos parisinos, hoy es el monumento más visitado del mundo.
        Mide 324 metros de altura y tiene tres niveles accesibles para visitantes.
        Cada noche, la torre ofrece un espectáculo de luces que dura cinco minutos cada hora.
        """),
        ("museos_paris.txt", """
        París alberga algunos de los museos más importantes del mundo.
        El Museo del Louvre es el más grande y famoso, hogar de la Mona Lisa y la Venus de Milo.
        El Museo de Orsay, ubicado en una antigua estación de tren, contiene una impresionante colección de arte impresionista.
        El Centro Pompidou es reconocido por su arquitectura moderna y su colección de arte contemporáneo.
        """)
    ]
    
    for filename, content in files:
        with open(os.path.join(temp_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)
    
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(test_indices_dir):
    """Fixture para crear una configuración de prueba."""
    config = Config()
    config.INDICES_DIR = Path(test_indices_dir)
    return config

@pytest.fixture
async def setup_system(test_config, test_docs_dir):
    """Fixture para configurar todo el sistema e ingestar documentos de prueba."""
    # Crear componentes
    document_reader = DocumentReader()
    document_processor = DocumentProcessor(test_config)
    chunk_repository = ChunkRepository(test_config)
    embedding_service = EmbeddingService(test_config)
    embedding_repository = EmbeddingRepository(test_config)
    faiss_store = FAISSVectorStore(test_config)
    sync_service = SyncService(test_config, faiss_store, embedding_repository)
    query_processor = QueryProcessor(test_config, embedding_service)
    retriever = Retriever(faiss_store, embedding_repository, test_config)
    context_builder = ContextBuilder(test_config, chunk_repository)
    
    # Leer y procesar documentos
    doc_ids = []
    files = os.listdir(test_docs_dir)
    
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(test_docs_dir, filename)
            doc_data = document_reader.read_document(file_path)
            
            # Crear documento en BD
            doc_id = await chunk_repository.create_document(
                title=filename,
                metadata={"source": "test", "format": "text"}
            )
            doc_ids.append(doc_id)
            
            # Procesar documento
            document_input = {
                'content': doc_data['content'],
                'metadata': {"source": "test", "format": "text"},
                'doc_id': doc_id,
                'title': filename
            }
            chunks = document_processor.create_chunks(document_input)
            
            # Añadir metadatos necesarios
            for i, chunk in enumerate(chunks):
                chunk['doc_id'] = doc_id
                chunk['chunk_number'] = i
            
            # Almacenar chunks
            chunk_ids = await chunk_repository.store_chunks(doc_id, chunks)
            
            # Generar y almacenar embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(texts)
            
            for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                await embedding_repository.store_embedding(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    model_name='miniLM'
                )
    
    # Sincronizar con FAISS
    await sync_service.synchronize()
    
    # Retornar componentes y datos creados
    components = {
        'document_reader': document_reader,
        'document_processor': document_processor,
        'chunk_repository': chunk_repository,
        'embedding_service': embedding_service,
        'embedding_repository': embedding_repository,
        'faiss_store': faiss_store,
        'sync_service': sync_service,
        'query_processor': query_processor,
        'retriever': retriever,
        'context_builder': context_builder
    }
    
    yield {
        'components': components,
        'doc_ids': doc_ids
    }
    
    # Limpiar
    for doc_id in doc_ids:
        await chunk_repository.delete_document(doc_id)
    
    await chunk_repository.close()
    await embedding_repository.close()

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_end_to_end_query_flow(setup_system):
    """Prueba el flujo completo desde consulta hasta contexto."""
    components = setup_system['components']
    
    # Componentes necesarios
    query_processor = components['query_processor']
    retriever = components['retriever']
    context_builder = components['context_builder']
    
    # Lista de consultas para probar
    test_queries = [
        "¿Qué museos hay en París?",
        "Información sobre la Torre Eiffel",
        "Historia de París"
    ]
    
    for query_text in test_queries:
        logger.info(f"\n\nProcesando consulta: '{query_text}'")
        
        # 1. Procesar consulta
        query_data = await query_processor.process_query(query_text)
        logger.info(f"Consulta procesada - Idioma: {query_data['language']}, Tipo: {query_data['query_type']}")
        
        # 2. Recuperar chunks relevantes
        retrieval_results = await retriever.retrieve(
            query_vector=query_data['embedding'],
            k=3,
            mode='faiss'
        )
        
        # Verificar resultados de recuperación
        assert 'indices' in retrieval_results or 'chunk_ids' in retrieval_results
        
        # 3. Construir contexto
        context_result = await context_builder.build_from_retrieval_results(
            retrieval_results=retrieval_results
        )
        
        # Verificar contexto
        assert 'context' in context_result
        assert len(context_result['context']) > 0
        
        logger.info(f"Contexto generado ({len(context_result['context'])} caracteres):")
        logger.info(f"------------------------------------------")
        logger.info(context_result['context'][:300] + "...")
        logger.info(f"------------------------------------------")
        
        # 4. Verificar chunks recuperados
        logger.info(f"Chunks recuperados: {len(context_result['chunks'])}")
        for i, chunk in enumerate(context_result['chunks']):
            logger.info(f"  Chunk {i+1}: {chunk['content']}")

def test_basic_rag_components(test_config):
    """Prueba la inicialización básica de los componentes del sistema RAG."""
    # Crear componentes
    document_reader = DocumentReader()
    document_processor = DocumentProcessor(test_config)
    embedding_service = EmbeddingService(test_config)
    faiss_store = FAISSVectorStore(test_config)
    query_processor = QueryProcessor(test_config, embedding_service)
    
    # Verificar que se inicializan correctamente
    assert document_reader is not None
    assert document_processor is not None
    assert embedding_service is not None
    assert faiss_store is not None
    assert query_processor is not None
    
    # Probar procesamiento básico
    test_text = "Este es un texto de prueba para el sistema RAG."
    
    # Normalización de consulta
    normalized = query_processor.normalize_query(test_text)
    assert normalized == test_text
    
    # Generar embedding
    embedding = embedding_service.generate_embeddings([test_text])
    assert len(embedding) == 1
    assert len(embedding[0]) > 0  # El embedding tiene dimensiones

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_different_retrieval_modes(setup_system):
    """Prueba diferentes modos de recuperación (FAISS, pgvector, híbrido)."""
    components = setup_system['components']
    
    # Componentes necesarios
    query_processor = components['query_processor']
    retriever = components['retriever']
    
    # Consulta de prueba
    query_text = "información sobre museos en París"
    
    # Procesar consulta
    query_data = await query_processor.process_query(query_text)
    
    # Probar modo FAISS
    faiss_results = await retriever.retrieve(
        query_vector=query_data['embedding'],
        k=2,
        mode='faiss'
    )
    logger.info(f"Modo FAISS - Resultados: {len(faiss_results.get('indices', [[]])[0])}")
    
    # Probar modo pgvector (skip si no está disponible)
    try:
        pgvector_results = await retriever.retrieve(
            query_vector=query_data['embedding'],
            k=2,
            mode='pgvector'
        )
        logger.info(f"Modo pgvector - Resultados: {len(pgvector_results.get('chunk_ids', []))}")
    except Exception as e:
        logger.warning(f"Modo pgvector no disponible: {e}")
    
    # Probar modo híbrido
    try:
        hybrid_results = await retriever.retrieve(
            query_vector=query_data['embedding'],
            k=2,
            mode='hybrid'
        )
        logger.info(f"Modo híbrido - Resultados: {len(hybrid_results.get('chunk_ids', []))}")
    except Exception as e:
        logger.warning(f"Modo híbrido no disponible: {e}")

@pytest.mark.asyncio
@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
async def test_context_building_strategies(setup_system):
    """Prueba diferentes estrategias de construcción de contexto."""
    components = setup_system['components']
    
    # Componentes necesarios
    query_processor = components['query_processor']
    retriever = components['retriever']
    context_builder = components['context_builder']
    
    # Consulta de prueba
    query_text = "historia y monumentos de París"
    
    # Recuperar chunks
    retrieval_results = await retriever.retrieve_by_text(
        query_processor=query_processor,
        query_text=query_text,
        k=4
    )
    
    # Verificar que hay resultados
    if not retrieval_results.get('chunk_ids'):
        pytest.skip("No se encontraron resultados para construir contexto")
    
    # Probar diferentes estrategias
    strategies = ['relevance', 'chronological', 'by_document']
    
    for strategy in strategies:
        logger.info(f"\nProbando estrategia: {strategy}")
        
        context_result = await context_builder.build_from_retrieval_results(
            retrieval_results=retrieval_results,
            strategy=strategy
        )
        
        logger.info(f"Contexto construido con {len(context_result['chunks'])} chunks")
        logger.info(f"Primeros 100 caracteres: {context_result['context'][:100]}...")
        
        # Verificar estructura y contenido
        assert context_result['strategy'] == strategy
        assert len(context_result['context']) > 0
        assert len(context_result['chunks']) > 0

def test_faiss_simple_operations():
    """Prueba operaciones simples con FAISS en un entorno aislado."""
    import numpy as np
    import tempfile
    import os
    from pathlib import Path
    
    # Crear un directorio temporal para el índice
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Crear configuración y FAISS store con el directorio temporal
        config = Config()
        config.INDICES_DIR = Path(temp_dir)
        faiss_store = FAISSVectorStore(config)
        
        # Verificar que el índice está vacío inicialmente
        initial_info = faiss_store.get_index_info()
        initial_vectors = initial_info['total_vectors']
        
        # Generar vectores de prueba
        test_vectors = np.random.rand(5, 384).astype(np.float32)
        
        # Añadir vectores
        ids = faiss_store.add_vectors(test_vectors)
        assert len(ids) == 5
        
        # Realizar búsqueda
        query = np.random.rand(1, 384).astype(np.float32)
        distances, indices = faiss_store.search(query, k=3)
        
        # Verificar resultados
        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)
        
        # Información del índice - verificar que se agregaron exactamente 5 vectores
        info = faiss_store.get_index_info()
        assert info['total_vectors'] == initial_vectors + 5
    
    finally:
        # Limpiar directorio temporal
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    pytest.main(["-v", "test_e2e.py"])