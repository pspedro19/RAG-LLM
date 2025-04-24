# app/core/db/service.py

import asyncpg
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config.config import get_settings

logger = logging.getLogger(__name__)

class DatabaseService:
   """
   Servicio base para gestionar operaciones de base de datos relacionadas con el sistema RAG.
   Proporciona conexión y funcionalidades comunes de base de datos.
   """
   
   def __init__(self):
       self.settings = get_settings()
       self.db_config = self.settings.get_db_config()
       self.pool = None
       
   async def initialize(self):
       """Inicializa el servicio de base de datos."""
       await self._get_pool()
       logger.info("Database service initialized")
       
   async def _get_pool(self):
       """Obtiene o crea el pool de conexiones asíncronas a PostgreSQL."""
       if self.pool is None:
           try:
               self.pool = await asyncpg.create_pool(
                   user=self.db_config.username,
                   password=self.db_config.password,
                   database=self.db_config.database,
                   host=self.db_config.host,
                   port=self.db_config.port
               )
               logger.info(f"Connected to PostgreSQL at {self.db_config.host}:{self.db_config.port}")
               
               # Configurar el decodificador de vectores
               async with self.pool.acquire() as conn:
                   await conn.set_type_codec(
                       'vector',
                       encoder=lambda v: v,
                       decoder=lambda v: [float(x) for x in v.strip()[1:-1].split(',')],
                       schema='public',
                       format='text'
                   )
           except Exception as e:
               logger.error(f"Error connecting to database: {e}")
               raise
       return self.pool
   
   async def execute_query(self, query: str, *args):
       """Ejecuta una consulta SQL genérica."""
       pool = await self._get_pool()
       async with pool.acquire() as conn:
           return await conn.execute(query, *args)
   
   async def fetch_all(self, query: str, *args):
       """Ejecuta una consulta y devuelve todas las filas."""
       pool = await self._get_pool()
       async with pool.acquire() as conn:
           rows = await conn.fetch(query, *args)
           return [dict(row) for row in rows]
   
   async def fetch_one(self, query: str, *args):
       """Ejecuta una consulta y devuelve una fila."""
       pool = await self._get_pool()
       async with pool.acquire() as conn:
           row = await conn.fetchrow(query, *args)
           return dict(row) if row else None
   
   async def fetch_val(self, query: str, *args):
       """Ejecuta una consulta y devuelve un valor único."""
       pool = await self._get_pool()
       async with pool.acquire() as conn:
           return await conn.fetchval(query, *args)
   
   async def close(self):
       """Cierra el pool de conexiones."""
       if self.pool:
           await self.pool.close()
           self.pool = None
           logger.info("Database connection pool closed")


class DocumentRepository:
   """Repositorio para operaciones relacionadas con documentos."""
   
   def __init__(self, db_service: DatabaseService):
       self.db = db_service
   
   async def create(self, title: str, metadata: Dict = None) -> str:
       """
       Crea un nuevo documento en la base de datos.
       
       Args:
           title: Título del documento
           metadata: Metadatos opcionales del documento
           
       Returns:
           str: ID del documento creado
       """
       doc_id = await self.db.fetch_val("""
           INSERT INTO documents (title, metadata)
           VALUES ($1, $2)
           RETURNING doc_id
       """, title, metadata or {})
       return str(doc_id)
   
   async def get_by_id(self, doc_id: str) -> Dict:
       """Obtiene un documento por su ID."""
       return await self.db.fetch_one("""
           SELECT doc_id, title, metadata, created_at
           FROM documents
           WHERE doc_id = $1
       """, doc_id)
   
   async def get_stats(self) -> int:
       """Obtiene el número total de documentos."""
       return await self.db.fetch_val("SELECT COUNT(*) FROM documents")


class ChunkRepository:
   """Repositorio para operaciones relacionadas con chunks."""
   
   def __init__(self, db_service: DatabaseService):
       self.db = db_service
   
   async def create_many(self, doc_id: str, chunks: List[Dict]) -> List[str]:
       """
       Crea múltiples chunks para un documento.
       
       Args:
           doc_id: ID del documento al que pertenecen los chunks
           chunks: Lista de diccionarios con información de chunks
           
       Returns:
           List[str]: Lista de IDs de chunks creados
       """
       chunk_ids = []
       
       for idx, chunk in enumerate(chunks):
           chunk_id = await self.db.fetch_val("""
               INSERT INTO chunks (doc_id, content, chunk_number, metadata)
               VALUES ($1, $2, $3, $4)
               RETURNING chunk_id
           """, doc_id, chunk['content'], idx, chunk.get('metadata', {}))
           chunk_ids.append(str(chunk_id))
           
       return chunk_ids
   
   async def get_by_document(self, doc_id: str) -> List[Dict]:
       """Obtiene todos los chunks de un documento."""
       return await self.db.fetch_all("""
           SELECT chunk_id, content, chunk_number, needs_indexing, created_at
           FROM chunks
           WHERE doc_id = $1
           ORDER BY chunk_number
       """, doc_id)
   
   async def get_pending(self, limit: int = 1000) -> List[Dict]:
       """Obtiene chunks pendientes de generar embeddings."""
       return await self.db.fetch_all("""
           SELECT c.chunk_id, c.content, c.doc_id, d.title
           FROM chunks c
           JOIN documents d ON c.doc_id = d.doc_id
           WHERE c.needs_indexing = TRUE
           ORDER BY c.created_at
           LIMIT $1
       """, limit)
   
   async def mark_as_indexed(self, chunk_id: str):
       """Marca un chunk como indexado."""
       await self.db.execute_query("""
           UPDATE chunks
           SET needs_indexing = FALSE
           WHERE chunk_id = $1
       """, chunk_id)
   
   async def get_stats(self) -> Dict:
       """Obtiene estadísticas de chunks."""
       total = await self.db.fetch_val("SELECT COUNT(*) FROM chunks")
       pending = await self.db.fetch_val("SELECT COUNT(*) FROM chunks WHERE needs_indexing = TRUE")
       
       return {
           'total': total,
           'pending_indexing': pending
       }


class EmbeddingRepository:
   """Repositorio para operaciones relacionadas con embeddings."""
   
   def __init__(self, db_service: DatabaseService):
       self.db = db_service
   
   async def create(
       self,
       chunk_id: str,
       model_name: str,
       vector: np.ndarray,
       faiss_index_id: Optional[int] = None
   ) -> str:
       """
       Inserta un nuevo embedding en la base de datos.
       
       Args:
           chunk_id: ID del chunk asociado
           model_name: Nombre del modelo de embedding
           vector: Vector de embedding como numpy array
           faiss_index_id: ID opcional en el índice FAISS
           
       Returns:
           str: ID del embedding creado
       """
       # Iniciar transacción para asegurar atomicidad
       pool = await self.db._get_pool()
       async with pool.acquire() as conn:
           async with conn.transaction():
               embedding_id = await conn.fetchval("""
                   INSERT INTO embeddings (chunk_id, model_name, embedding, faiss_index_id)
                   VALUES ($1, $2, $3, $4)
                   RETURNING embedding_id
               """, chunk_id, model_name, vector.tolist(), faiss_index_id)
               
               # Marcar el chunk como indexado
               await conn.execute("""
                   UPDATE chunks
                   SET needs_indexing = FALSE
                   WHERE chunk_id = $1
               """, chunk_id)
               
               return str(embedding_id)
   
   async def get_unsynchronized(self, limit: int = 10000) -> List[Dict[str, Any]]:
       """
       Obtiene embeddings que no están sincronizados con FAISS.
       
       Args:
           limit: Número máximo de embeddings a obtener
           
       Returns:
           Lista de embeddings pendientes de sincronizar
       """
       rows = await self.db.fetch_all("""
           SELECT embedding_id, embedding, chunk_id, e.created_at, c.content, d.title
           FROM embeddings e
           JOIN chunks c ON e.chunk_id = c.chunk_id
           JOIN documents d ON c.doc_id = d.doc_id
           WHERE e.faiss_index_id IS NULL
           ORDER BY e.created_at
           LIMIT $1
       """, limit)
       
       return [
           {
               'embedding_id': str(row['embedding_id']),
               'embedding': np.array(row['embedding'], dtype=np.float32),
               'chunk_id': str(row['chunk_id']),
               'content': row['content'],
               'title': row['title'],
               'created_at': row['created_at']
           }
           for row in rows
       ]
   
   async def update_faiss_ids(self, id_pairs: List[Tuple[str, int]]):
       """
       Actualiza los IDs de FAISS en la base de datos.
       
       Args:
           id_pairs: Lista de tuplas (embedding_id, faiss_index_id)
       """
       pool = await self.db._get_pool()
       async with pool.acquire() as conn:
           await conn.executemany("""
               UPDATE embeddings
               SET faiss_index_id = $2
               WHERE embedding_id = $1
           """, id_pairs)
           
       logger.info(f"Updated FAISS IDs for {len(id_pairs)} embeddings")
   
   async def get_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[Dict]:
       """
       Obtiene chunks por IDs de FAISS.
       
       Args:
           faiss_ids: Lista de IDs de FAISS
           
       Returns:
           Lista de chunks con su contenido y metadatos
       """
       return await self.db.fetch_all("""
           SELECT c.chunk_id, c.content, c.metadata, d.title, d.doc_id
           FROM embeddings e
           JOIN chunks c ON e.chunk_id = c.chunk_id
           JOIN documents d ON c.doc_id = d.doc_id
           WHERE e.faiss_index_id = ANY($1::int[])
           ORDER BY array_position($1::int[], e.faiss_index_id::int)
       """, faiss_ids)
   
   async def get_stats(self) -> Dict:
       """Obtiene estadísticas de embeddings."""
       total = await self.db.fetch_val("SELECT COUNT(*) FROM embeddings")
       unsynchronized = await self.db.fetch_val("SELECT COUNT(*) FROM embeddings WHERE faiss_index_id IS NULL")
       
       return {
           'total': total,
           'unsynchronized': unsynchronized
       }


class RAGDatabaseService:
   """
   Servicio unificado para gestionar todas las operaciones de base de datos del sistema RAG.
   Proporciona acceso a todos los repositorios.
   """
   
   def __init__(self):
       self.db_service = DatabaseService()
       self.documents = DocumentRepository(self.db_service)
       self.chunks = ChunkRepository(self.db_service)
       self.embeddings = EmbeddingRepository(self.db_service)
   
   async def initialize(self):
       """Inicializa el servicio y todos sus repositorios."""
       await self.db_service.initialize()
       logger.info("RAG Database Service initialized")
   
   async def get_stats(self) -> Dict:
       """Obtiene estadísticas completas del sistema RAG."""
       docs_count = await self.db_service.fetch_val("SELECT COUNT(*) FROM documents")
       chunks_count = await self.db_service.fetch_val("SELECT COUNT(*) FROM chunks")
       embeddings_count = await self.db_service.fetch_val("SELECT COUNT(*) FROM embeddings")
       pending_count = await self.db_service.fetch_val("SELECT COUNT(*) FROM chunks WHERE needs_indexing = TRUE")
       unsync_count = await self.db_service.fetch_val("SELECT COUNT(*) FROM embeddings WHERE faiss_index_id IS NULL")
       
       return {
           'documents_count': docs_count,
           'chunks_count': chunks_count,
           'embeddings_count': embeddings_count,
           'pending_indexing': pending_count,
           'unsynchronized': unsync_count
       }
   
   async def close(self):
       """Cierra las conexiones de base de datos."""
       await self.db_service.close()
       logger.info("RAG Database Service closed")


# Singleton instance for global access
_rag_db_service = None

async def get_rag_db_service() -> RAGDatabaseService:
   """
   Obtiene o crea una instancia singleton del servicio de base de datos RAG.
   """
   global _rag_db_service
   
   if _rag_db_service is None:
       _rag_db_service = RAGDatabaseService()
       await _rag_db_service.initialize()
   
   return _rag_db_service