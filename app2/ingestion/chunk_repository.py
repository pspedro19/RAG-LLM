# app2/ingestion/chunk_repository.py

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import json
import asyncpg
from datetime import datetime

from app2.core.config.config import Config

logger = logging.getLogger(__name__)

class ChunkRepository:
    """
    Repositorio para interactuar con la tabla de chunks en PostgreSQL.
    Maneja la persistencia y recuperación de chunks de documentos.
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el repositorio de chunks.
        
        Args:
            config: Configuración con credenciales de BD
        """
        self.config = config
        self.pool = None
    
    async def _get_pool(self):
        """
        Obtiene o crea el pool de conexiones a PostgreSQL.
        
        Returns:
            Pool de conexiones asyncpg
        """
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT
            )
        return self.pool
    
    async def close(self):
        """Cierra el pool de conexiones si existe."""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Recupera un documento por su ID.
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Datos del documento o None si no existe
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            record = await conn.fetchrow("""
                SELECT doc_id, title, metadata, created_at
                FROM documents
                WHERE doc_id = $1
            """, doc_id)
            
            if not record:
                return None
                
            return {
                'doc_id': record['doc_id'],
                'title': record['title'],
                'metadata': record['metadata'] if isinstance(record['metadata'], dict) else json.loads(record['metadata']),
                'created_at': record['created_at'].isoformat()
            }
    
    async def create_document(self, title: str, metadata: Dict = None) -> str:
        """
        Crea un nuevo documento en la base de datos.
        
        Args:
            title: Título del documento
            metadata: Metadatos opcionales
            
        Returns:
            ID del documento creado
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            doc_id = await conn.fetchval("""
                INSERT INTO documents (title, metadata)
                VALUES ($1, $2)
                RETURNING doc_id
            """, title, json.dumps(metadata or {}))
            
            return str(doc_id)
    
    async def update_document(self, doc_id: str, title: str = None, metadata: Dict = None) -> bool:
        """
        Actualiza un documento existente.
        
        Args:
            doc_id: ID del documento
            title: Nuevo título (opcional)
            metadata: Nuevos metadatos (opcional)
            
        Returns:
            True si se actualizó correctamente
        """
        if title is None and metadata is None:
            return True  # Nada que actualizar
        
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Construir consulta según los campos a actualizar
            updates = []
            params = [doc_id]
            
            if title is not None:
                updates.append(f"title = ${len(params) + 1}")
                params.append(title)
            
            if metadata is not None:
                updates.append(f"metadata = ${len(params) + 1}")
                params.append(json.dumps(metadata))
            
            query = f"""
                UPDATE documents
                SET {', '.join(updates)}
                WHERE doc_id = $1
                RETURNING doc_id
            """
            
            updated_id = await conn.fetchval(query, *params)
            return updated_id is not None
    
    async def store_chunks(self, doc_id: str, chunks: List[Dict]) -> List[str]:
        """
        Almacena múltiples chunks para un documento.
        
        Args:
            doc_id: ID del documento
            chunks: Lista de chunks a almacenar
            
        Returns:
            Lista de IDs de chunks creados
        """
        if not chunks:
            return []
        
        # Verificar que el documento existe
        doc = await self.get_document_by_id(doc_id)
        if not doc:
            raise ValueError(f"El documento con ID {doc_id} no existe")
        
        pool = await self._get_pool()
        chunk_ids = []
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, chunk in enumerate(chunks):
                    # Generar ID de chunk si no tiene
                    chunk_id = str(uuid.uuid4())
                    
                    # Combinar metadatos específicos del chunk
                    metadata = {
                        **(chunk.get('metadata', {})),
                        'start': chunk.get('start'),
                        'end': chunk.get('end'),
                        'token_count': chunk.get('token_count'),
                        'hash': chunk.get('hash')
                    }
                    
                    # Insertar chunk
                    await conn.execute("""
                        INSERT INTO chunks 
                        (chunk_id, doc_id, content, chunk_number, metadata, needs_indexing)
                        VALUES ($1, $2, $3, $4, $5, TRUE)
                    """, 
                    chunk_id, 
                    doc_id, 
                    chunk['content'], 
                    chunk.get('chunk_number', i),
                    json.dumps(metadata))
                    
                    chunk_ids.append(chunk_id)
        
        logger.info(f"Almacenados {len(chunk_ids)} chunks para documento {doc_id}")
        return chunk_ids
    
    async def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict]:
        """
        Recupera todos los chunks de un documento.
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Lista de chunks con su contenido y metadatos
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chunk_id, content, chunk_number, metadata, created_at, needs_indexing
                FROM chunks
                WHERE doc_id = $1
                ORDER BY chunk_number
            """, doc_id)
            
            return [
                {
                    'chunk_id': row['chunk_id'],
                    'content': row['content'],
                    'chunk_number': row['chunk_number'],
                    'metadata': row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']),
                    'created_at': row['created_at'].isoformat(),
                    'needs_indexing': row['needs_indexing']
                }
                for row in rows
            ]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Recupera un chunk específico por su ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            Datos del chunk o None si no existe
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT c.chunk_id, c.doc_id, c.content, c.chunk_number, 
                       c.metadata, c.created_at, c.needs_indexing, d.title
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_id = $1
            """, chunk_id)
            
            if not row:
                return None
                
            return {
                'chunk_id': row['chunk_id'],
                'doc_id': row['doc_id'],
                'title': row['title'],
                'content': row['content'],
                'chunk_number': row['chunk_number'],
                'metadata': row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']),
                'created_at': row['created_at'].isoformat(),
                'needs_indexing': row['needs_indexing']
            }
    
    async def get_pending_chunks(self, limit: int = 1000) -> List[Dict]:
        """
        Recupera chunks pendientes de indexación (needs_indexing = TRUE).
        
        Args:
            limit: Número máximo de chunks a recuperar
            
        Returns:
            Lista de chunks pendientes
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.chunk_id, c.doc_id, c.content, c.chunk_number, 
                       c.metadata, d.title
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.needs_indexing = TRUE
                ORDER BY c.created_at
                LIMIT $1
            """, limit)
            
            return [
                {
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'chunk_number': row['chunk_number'],
                    'metadata': row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])
                }
                for row in rows
            ]
    
    async def mark_chunks_as_indexed(self, chunk_ids: List[str]) -> int:
        """
        Marca chunks como indexados (needs_indexing = FALSE).
        
        Args:
            chunk_ids: Lista de IDs de chunks a marcar
            
        Returns:
            Número de chunks actualizados
        """
        if not chunk_ids:
            return 0
            
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE chunks
                SET needs_indexing = FALSE
                WHERE chunk_id = ANY($1::uuid[])
            """, chunk_ids)
            
            # Extraer número de filas afectadas
            affected = int(result.split()[-1])
            return affected
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Elimina un documento y todos sus chunks.
        
        Args:
            doc_id: ID del documento a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # PostgreSQL eliminará automáticamente los chunks por la restricción ON DELETE CASCADE
                deleted = await conn.fetchval("""
                    DELETE FROM documents
                    WHERE doc_id = $1
                    RETURNING doc_id
                """, doc_id)
                
                return deleted is not None