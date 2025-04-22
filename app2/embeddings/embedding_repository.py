# app2/embeddings/embedding_repository.py

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import numpy as np
import asyncpg
from datetime import datetime

from app2.core.config.config import Config

logger = logging.getLogger(__name__)

class EmbeddingRepository:
    """
    Repositorio para gestionar embeddings en PostgreSQL.
    Maneja almacenamiento, recuperación y sincronización con FAISS.
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el repositorio de embeddings.
        
        Args:
            config: Configuración con credenciales de BD
        """
        self.config = config
        self.pool = None
    
    async def _get_pool(self):
        """
        Obtiene o crea el pool de conexiones a PostgreSQL.
        Configura también el codec para manejar vectores.
        
        Returns:
            Pool de conexiones asyncpg
        """
        if self.pool is None:
            async def init_connection(conn):
                await conn.set_type_codec(
                    'vector',
                    encoder=lambda v: v,
                    decoder=lambda v: [float(x) for x in v.strip()[1:-1].split(',')],
                    schema='public',
                    format='text'
                )
            
            self.pool = await asyncpg.create_pool(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                init=init_connection
            )
        return self.pool
    
    async def close(self):
        """Cierra el pool de conexiones si existe."""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def store_embedding(
        self,
        chunk_id: str,
        embedding: np.ndarray,
        model_name: str = 'miniLM',
        faiss_index_id: Optional[int] = None
    ) -> str:
        """
        Almacena un embedding para un chunk específico.
        
        Args:
            chunk_id: ID del chunk asociado
            embedding: Vector de embedding (numpy array)
            model_name: Nombre del modelo usado
            faiss_index_id: ID opcional en el índice FAISS
            
        Returns:
            ID del embedding creado
            
        Raises:
            ValueError: Si hay problemas con el formato del embedding
        """
        # Validar embedding
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Asegurar que sea vector fila 1D
        if embedding.ndim > 1:
            if embedding.shape[0] == 1:
                embedding = embedding[0]
            else:
                raise ValueError(f"Formato incorrecto de embedding: shape={embedding.shape}")
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Verificar que el chunk existe
            chunk_exists = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE chunk_id = $1", 
                chunk_id
            )
            
            if not chunk_exists:
                raise ValueError(f"El chunk con ID {chunk_id} no existe")
            
            # Convertir el embedding a formato de texto para pgvector (usando corchetes, no llaves)
            # Pgvector requiere formato [x1,x2,x3,...] en lugar de {x1,x2,x3,...}
            embedding_str = f"[{','.join(str(float(x)) for x in embedding)}]"
            
            # Insertar embedding
            embedding_id = await conn.fetchval("""
                INSERT INTO embeddings (chunk_id, model_name, embedding, faiss_index_id)
                VALUES ($1, $2, $3::vector, $4)
                RETURNING embedding_id
            """, chunk_id, model_name, embedding_str, faiss_index_id)
            
            return str(embedding_id)
    
    async def store_batch_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        model_name: str = 'miniLM',
        faiss_index_ids: Optional[List[int]] = None
    ) -> List[str]:
        """
        Almacena múltiples embeddings en modo batch.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            embeddings: Matriz de embeddings (numpy array 2D)
            model_name: Nombre del modelo usado
            faiss_index_ids: Lista opcional de IDs en el índice FAISS
            
        Returns:
            Lista de IDs de embeddings creados
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(f"Número incorrecto de embeddings: {len(chunk_ids)} != {len(embeddings)}")
        
        if not faiss_index_ids:
            faiss_index_ids = [None] * len(chunk_ids)
        
        pool = await self._get_pool()
        embedding_ids = []
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, (chunk_id, embedding, faiss_id) in enumerate(zip(chunk_ids, embeddings, faiss_index_ids)):
                    # Asegurar que embedding sea 1D
                    if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
                        embedding = embedding[0]
                    
                    # Convertir el embedding a formato de texto para pgvector usando corchetes
                    embedding_str = f"[{','.join(str(float(x)) for x in embedding)}]"
                    
                    embedding_id = await conn.fetchval("""
                        INSERT INTO embeddings (chunk_id, model_name, embedding, faiss_index_id)
                        VALUES ($1, $2, $3::vector, $4)
                        RETURNING embedding_id
                    """, chunk_id, model_name, embedding_str, faiss_id)
                    
                    embedding_ids.append(str(embedding_id))
        
        logger.info(f"Almacenados {len(embedding_ids)} embeddings en batch")
        return embedding_ids
    
    async def get_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera un embedding específico por su ID.
        
        Args:
            embedding_id: ID del embedding
            
        Returns:
            Diccionario con embedding y metadatos o None si no existe
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    e.embedding_id, e.chunk_id, e.model_name, 
                    e.embedding, e.faiss_index_id, e.created_at,
                    c.content, d.title
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE e.embedding_id = $1
            """, embedding_id)
            
            if not row:
                return None
            
            return {
                'embedding_id': row['embedding_id'],
                'chunk_id': row['chunk_id'],
                'model_name': row['model_name'],
                'embedding': np.array(row['embedding'], dtype=np.float32),
                'faiss_index_id': row['faiss_index_id'],
                'created_at': row['created_at'].isoformat(),
                'content': row['content'],
                'title': row['title']
            }
    
    async def get_embedding_by_chunk(
        self, 
        chunk_id: str, 
        model_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Recupera el embedding asociado a un chunk específico.
        
        Args:
            chunk_id: ID del chunk
            model_name: Nombre opcional del modelo (si hay múltiples por chunk)
            
        Returns:
            Diccionario con embedding y metadatos o None si no existe
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    e.embedding_id, e.chunk_id, e.model_name, 
                    e.embedding, e.faiss_index_id, e.created_at
                FROM embeddings e
                WHERE e.chunk_id = $1
            """
            
            params = [chunk_id]
            
            if model_name:
                query += " AND e.model_name = $2"
                params.append(model_name)
            
            row = await conn.fetchrow(query, *params)
            
            if not row:
                return None
            
            return {
                'embedding_id': row['embedding_id'],
                'chunk_id': row['chunk_id'],
                'model_name': row['model_name'],
                'embedding': np.array(row['embedding'], dtype=np.float32),
                'faiss_index_id': row['faiss_index_id'],
                'created_at': row['created_at'].isoformat()
            }
    
    async def get_embeddings_for_chunks(
        self, 
        chunk_ids: List[str], 
        model_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera embeddings para múltiples chunks.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            model_name: Nombre opcional del modelo
            
        Returns:
            Lista de diccionarios con embeddings y metadatos
        """
        if not chunk_ids:
            return []
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    e.embedding_id, e.chunk_id, e.model_name, 
                    e.embedding, e.faiss_index_id, e.created_at
                FROM embeddings e
                WHERE e.chunk_id = ANY($1::uuid[])
            """
            
            params = [chunk_ids]
            
            if model_name:
                query += " AND e.model_name = $2"
                params.append(model_name)
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    'embedding_id': row['embedding_id'],
                    'chunk_id': row['chunk_id'],
                    'model_name': row['model_name'],
                    'embedding': np.array(row['embedding'], dtype=np.float32),
                    'faiss_index_id': row['faiss_index_id'],
                    'created_at': row['created_at'].isoformat()
                }
                for row in rows
            ]
    
    async def get_unsynchronized_embeddings(
        self, 
        limit: int = 1000, 
        model_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera embeddings que aún no están sincronizados con FAISS.
        
        Args:
            limit: Número máximo de embeddings a recuperar
            model_name: Filtro opcional por modelo
            
        Returns:
            Lista de diccionarios con embeddings pendientes
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    e.embedding_id, e.chunk_id, e.model_name, 
                    e.embedding, e.created_at,
                    c.content, d.title
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE e.faiss_index_id IS NULL
            """
            
            params = []
            
            if model_name:
                query += " AND e.model_name = $1"
                params.append(model_name)
            
            query += f" ORDER BY e.created_at LIMIT {limit}"
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    'embedding_id': row['embedding_id'],
                    'chunk_id': row['chunk_id'],
                    'model_name': row['model_name'],
                    'embedding': np.array(row['embedding'], dtype=np.float32),
                    'created_at': row['created_at'].isoformat(),
                    'content': row['content'],
                    'title': row['title']
                }
                for row in rows
            ]
    
    async def update_faiss_ids(self, id_pairs: List[Tuple[str, int]]) -> int:
        """
        Actualiza los IDs de FAISS para embeddings específicos.
        
        Args:
            id_pairs: Lista de tuplas (embedding_id, faiss_index_id)
            
        Returns:
            Número de embeddings actualizados
        """
        if not id_pairs:
            return 0
        
        pool = await self._get_pool()
        count = 0
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                for emb_id, faiss_id in id_pairs:
                    result = await conn.execute("""
                        UPDATE embeddings
                        SET faiss_index_id = $2
                        WHERE embedding_id = $1
                    """, emb_id, faiss_id)
                    
                    # Sumar filas afectadas
                    count += int(result.split()[-1])
        
        logger.info(f"Actualizados {count} IDs FAISS en embeddings")
        return count
    
    async def mark_chunks_indexed(self, chunk_ids: List[str]) -> int:
        """
        Marca chunks como indexados en FAISS.
        
        Args:
            chunk_ids: Lista de IDs de chunks a marcar como indexados
            
        Returns:
            Número de chunks actualizados
        """
        if not chunk_ids:
            return 0
        
        pool = await self._get_pool()
        count = 0
        
        try:
            async with pool.acquire() as conn:
                # Marcar chunks como indexados, si existe el campo needs_indexing
                try:
                    result = await conn.execute("""
                        UPDATE chunks
                        SET needs_indexing = FALSE
                        WHERE chunk_id = ANY($1::uuid[])
                    """, chunk_ids)
                    
                    # Contar filas afectadas si es posible
                    if result and ' ' in result:
                        count = int(result.split()[-1])
                except Exception as e:
                    # Si el campo no existe, simplemente lo ignoramos y continuamos
                    logger.debug(f"No se pudo marcar chunks como indexados: {e}")
        except Exception as e:
            logger.error(f"Error al marcar chunks como indexados: {e}")
        
        logger.info(f"Marcados {count} chunks como indexados")
        return count
    
    async def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        distance_threshold: float = 0.3,  # Lowered from 0.5 to 0.3 to be even less restrictive
        model_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Busca embeddings similares usando pgvector.
        Nota: Requiere extensión pgvector instalada y configurada.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados a retornar
            distance_threshold: Umbral de similitud (coseno), valor más bajo encuentra más resultados
            model_name: Filtro opcional por modelo
            
        Returns:
            Lista de resultados ordenados por similitud
        """
        # Validar vector de consulta
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        if query_vector.ndim > 1 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Comprobar si existe la extensión pgvector
            has_pgvector = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            )
            
            if not has_pgvector:
                raise RuntimeError("La extensión pgvector no está instalada en PostgreSQL")
            
            # Convertir el vector de consulta a formato de texto para pgvector (usando corchetes)
            query_vector_str = f"[{','.join(str(float(x)) for x in query_vector)}]"
            
            # Construir consulta
            query = """
                SELECT 
                    e.embedding_id, e.chunk_id, e.model_name,
                    1 - (e.embedding <=> $1::vector) AS similarity,
                    c.content, c.metadata, d.title
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE 1 - (e.embedding <=> $1::vector) > $2
            """
            
            params = [query_vector_str, distance_threshold]
            
            if model_name:
                query += " AND e.model_name = $3"
                params.append(model_name)
            
            query += f" ORDER BY similarity DESC LIMIT {k}"
            
            rows = await conn.fetch(query, *params)
            
            # Log similarity scores for debugging
            for i, row in enumerate(rows):
                logger.debug(f"Search result #{i+1}: Similarity: {row['similarity']:.4f}, " +
                           f"Title: {row['title']}, Content: {row['content'][:50]}...")
            
            return [
                {
                    'embedding_id': row['embedding_id'],
                    'chunk_id': row['chunk_id'],
                    'model_name': row['model_name'],
                    'similarity': row['similarity'],
                    'content': row['content'],
                    'metadata': row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']) if row['metadata'] else {},
                    'title': row['title']
                }
                for row in rows
            ]
    
    async def search_by_keywords(
        self,
        keywords: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por palabras clave en texto cuando la búsqueda semántica falla.
        
        Args:
            keywords: Palabras clave para buscar
            limit: Número máximo de resultados
            
        Returns:
            Lista de resultados relevantes
        """
        pool = await self._get_pool()
        
        # Preparar palabras clave para búsqueda
        # Eliminar palabras comunes y convertir a formato para búsqueda
        search_terms = keywords.lower().replace('?', '').replace('¿', '')
        search_terms = ' & '.join([term.strip() for term in search_terms.split() if len(term) > 3])
        
        if not search_terms:
            return []
            
        async with pool.acquire() as conn:
            try:
                # Intentar búsqueda por texto completo si está disponible
                rows = await conn.fetch("""
                    SELECT 
                        c.chunk_id, c.content, d.title,
                        ts_rank_cd(to_tsvector('spanish', c.content), to_tsquery('spanish', $1)) AS rank
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE to_tsvector('spanish', c.content) @@ to_tsquery('spanish', $1)
                    ORDER BY rank DESC
                    LIMIT $2
                """, search_terms, limit)
                
                if not rows:
                    # Búsqueda alternativa por ILIKE
                    keywords_list = [f"%{term.strip()}%" for term in keywords.lower().split() if len(term) > 3]
                    if not keywords_list:
                        return []
                    
                    query = """
                        SELECT 
                            c.chunk_id, c.content, d.title
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        WHERE 
                    """
                    conditions = []
                    params = []
                    
                    for i, term in enumerate(keywords_list):
                        conditions.append(f"LOWER(c.content) ILIKE ${i+1}")
                        params.append(term)
                    
                    query += " OR ".join(conditions)
                    query += f" LIMIT {limit}"
                    
                    rows = await conn.fetch(query, *params)
            except Exception as e:
                logger.warning(f"Error en búsqueda por palabras clave: {e}")
                # Último intento con una búsqueda simple
                keyword = f"%{keywords.split()[0] if keywords.split() else ''}%"
                rows = await conn.fetch("""
                    SELECT 
                        c.chunk_id, c.content, d.title
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE LOWER(c.content) ILIKE $1
                    LIMIT $2
                """, keyword, limit)
            
            return [
                {
                    'chunk_id': row['chunk_id'],
                    'content': row['content'],
                    'title': row['title'],
                    'similarity': row.get('rank', 0.5),  # Valor predeterminado si no hay rank
                }
                for row in rows
            ]
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre los embeddings almacenados.
        
        Returns:
            Diccionario con estadísticas
        """
        pool = await self._get_pool()
        stats = {}
        
        async with pool.acquire() as conn:
            # Total de embeddings
            stats['total_embeddings'] = await conn.fetchval(
                "SELECT COUNT(*) FROM embeddings"
            )
            
            # Distribución por modelo
            model_rows = await conn.fetch("""
                SELECT model_name, COUNT(*) as count
                FROM embeddings
                GROUP BY model_name
            """)
            
            stats['models'] = {
                row['model_name']: row['count']
                for row in model_rows
            }
            
            # Embeddings sincronizados vs no sincronizados
            synced = await conn.fetchval(
                "SELECT COUNT(*) FROM embeddings WHERE faiss_index_id IS NOT NULL"
            )
            
            stats['synced'] = synced
            stats['unsynced'] = stats['total_embeddings'] - synced
            
            # Total de documentos y chunks con embeddings
            stats['documents_with_embeddings'] = await conn.fetchval("""
                SELECT COUNT(DISTINCT d.doc_id)
                FROM documents d
                JOIN chunks c ON d.doc_id = c.doc_id
                JOIN embeddings e ON c.chunk_id = e.chunk_id
            """)
            
            stats['chunks_with_embeddings'] = await conn.fetchval("""
                SELECT COUNT(DISTINCT c.chunk_id)
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
            """)
        
        return stats
    
    async def get_total_embeddings_count(self) -> int:
        """
        Retorna el número total de embeddings en la base de datos.
        
        Returns:
            int: Número total de embeddings
        """
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
            return count or 0