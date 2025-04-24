# app/core/rag/sync/service.py
import asyncio
import logging
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime

from app.core.config.config import get_settings
from app.core.rag.vector_store.postgres import PostgresVectorStore
from app.core.rag.vector_store.faiss import get_faiss_manager

logger = logging.getLogger(__name__)

class SyncService:
   """
   Servicio de sincronización entre PostgreSQL (pgvector) y FAISS.
   
   Este servicio se encarga de mantener los embeddings de PostgreSQL
   sincronizados con el índice FAISS, para garantizar consistencia en
   las búsquedas vectoriales.
   """
   
   def __init__(self, pg_store: Optional[PostgresVectorStore] = None, faiss_manager = None):
       """
       Inicializa el servicio de sincronización.
       
       Args:
           pg_store: Instancia de PostgresVectorStore. Si es None, se crea automáticamente.
           faiss_manager: Instancia del gestor FAISS. Si es None, se obtiene automáticamente.
       """
       self.settings = get_settings()
       self._sync_task = None
       self.is_syncing = False
       self.is_scheduler_running = False
       self.last_sync_time = None
       
       # Instancias para manejo de almacenes vectoriales
       self.pg_store = pg_store or PostgresVectorStore()
       self.faiss_manager = faiss_manager
       
       # Estadísticas de sincronización
       self.sync_stats = {
           'total_processed': 0,
           'total_failed': 0,
           'last_batch_size': 0,
           'last_duration': 0,
       }
   
   async def get_faiss_manager(self):
       """
       Obtiene la instancia de FAISS Manager.
       """
       if self.faiss_manager is None:
           self.faiss_manager = await get_faiss_manager()
       return self.faiss_manager
   
   def _parse_embedding(self, embedding) -> np.ndarray:
       """
       Parsea un embedding de PostgreSQL a formato numpy compatible con FAISS.
       
       Args:
           embedding: Embedding en formato string, lista o array de PostgreSQL
           
       Returns:
           np.ndarray: Array de numpy con shape (1, dimension) y tipo float32
           
       Raises:
           ValueError: Si el embedding no puede convertirse correctamente
       """
       faiss_manager = self.faiss_manager
       
       if isinstance(embedding, (list, np.ndarray)):
           vector = np.array(embedding, dtype=np.float32)
       elif isinstance(embedding, str):
           # Formato común de pgvector: [0.1, 0.2, 0.3, ...]
           cleaned = embedding.strip('[').strip(']')
           try:
               numbers = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
               vector = np.array(numbers, dtype=np.float32)
           except Exception as e:
               raise ValueError(f"Error converting embedding to float: {e}")
       else:
           raise ValueError(f"Unsupported embedding type: {type(embedding)}")
       
       # Asegurar dimensionalidad correcta
       if vector.ndim == 1:
           vector = vector.reshape(1, -1)
       
       # Verificaciones de seguridad
       if faiss_manager and vector.shape[1] != faiss_manager.dimension:
           raise ValueError(f"Incorrect dimension: {vector.shape[1]} (expected {faiss_manager.dimension})")
       
       if not np.all(np.isfinite(vector)):
           raise ValueError("Vector contains non-finite values (NaN or inf)")
       
       return vector
   
   async def synchronize(self, batch_size: int = 1000) -> Tuple[int, int]:
       """
       Sincroniza los embeddings entre PostgreSQL y FAISS.
       
       Args:
           batch_size: Tamaño del lote para procesar por iteración
           
       Returns:
           Tuple[int, int]: (procesados, fallidos)
       """
       if self.is_syncing:
           logger.warning("Synchronization already in progress")
           return 0, 0
       
       self.is_syncing = True
       processed = 0
       failed = 0
       
       try:
           start_time = datetime.now()
           faiss_manager = await self.get_faiss_manager()
           
           # Obtener embeddings pendientes de sincronización
           embeddings = await self.pg_store.get_unsynchronized_embeddings(limit=batch_size)
           
           if not embeddings:
               logger.info("No embeddings to synchronize")
               return 0, 0
           
           # Procesar embeddings en lotes para arrays grandes
           vectors = []
           embedding_ids = []
           chunk_ids = []
           
           for emb in embeddings:
               try:
                   vector = self._parse_embedding(emb['embedding'])
                   vectors.append(vector)
                   embedding_ids.append(emb['embedding_id'])
                   chunk_ids.append(emb['chunk_id'])
               except Exception as e:
                   logger.error(f"Error processing embedding {emb['embedding_id']}: {e}")
                   failed += 1
           
           if vectors:
               # Concatenar vectores para añadir a FAISS
               vectors_array = np.vstack(vectors)
               logger.info(f"Prepared {len(vectors)} vectors for FAISS with shape {vectors_array.shape}")
               
               try:
                   # Añadir vectores a FAISS
                   faiss_ids = faiss_manager.add_vectors(vectors_array)
                   
                   # Actualizar IDs de FAISS en PostgreSQL
                   id_mappings = [
                       (emb_id, faiss_id)
                       for emb_id, faiss_id in zip(embedding_ids, faiss_ids)
                   ]
                   await self.pg_store.update_faiss_ids(id_mappings)
                   
                   # Marcar chunks como indexados
                   await self.pg_store.mark_chunks_indexed(chunk_ids)
                   
                   processed = len(vectors)
                   logger.info(f"Synchronization successful: {processed} embeddings processed")
                   
               except Exception as e:
                   logger.error(f"Error processing vectors in FAISS: {e}")
                   failed += len(vectors)
           
           # Actualizar estadísticas
           end_time = datetime.now()
           self.sync_stats['total_processed'] += processed
           self.sync_stats['total_failed'] += failed
           self.sync_stats['last_batch_size'] = processed + failed
           self.sync_stats['last_duration'] = (end_time - start_time).total_seconds()
           self.last_sync_time = end_time
           
       except Exception as e:
           logger.error(f"Error during synchronization: {e}")
           raise
       finally:
           self.is_syncing = False
       
       return processed, failed
   
   async def full_synchronize(self) -> Tuple[int, int]:
       """
       Realiza una sincronización completa (reinicia el índice FAISS).
       
       Returns:
           Tuple[int, int]: (procesados, fallidos)
       """
       if self.is_syncing:
           logger.warning("Synchronization already in progress")
           return 0, 0
       
       self.is_syncing = True
       processed = 0
       failed = 0
       
       try:
           faiss_manager = await self.get_faiss_manager()
           
           # Crear backup del índice actual
           backup_path = faiss_manager.backup_index()
           if backup_path:
               logger.info(f"Backup created at {backup_path}")
           
           # Reiniciar índice FAISS
           faiss_manager.reset_index()
           
           # Obtener todos los embeddings
           embeddings = await self.pg_store.get_all_embeddings()
           
           if not embeddings:
               logger.info("No embeddings found for full synchronization")
               return 0, 0
           
           # Procesar todos los embeddings
           vectors = []
           embedding_ids = []
           chunk_ids = []
           
           for emb in embeddings:
               try:
                   vector = self._parse_embedding(emb['embedding'])
                   vectors.append(vector)
                   embedding_ids.append(emb['embedding_id'])
                   chunk_ids.append(emb['chunk_id'])
               except Exception as e:
                   logger.error(f"Error processing embedding {emb['embedding_id']}: {e}")
                   failed += 1
           
           if vectors:
               vectors_array = np.vstack(vectors)
               logger.info(f"Prepared {len(vectors)} vectors for full FAISS sync")
               
               try:
                   # Añadir vectores a FAISS
                   faiss_ids = faiss_manager.add_vectors(vectors_array)
                   
                   # Actualizar IDs de FAISS en PostgreSQL
                   id_mappings = [
                       (emb_id, faiss_id)
                       for emb_id, faiss_id in zip(embedding_ids, faiss_ids)
                   ]
                   await self.pg_store.update_faiss_ids(id_mappings)
                   
                   # Marcar todos los chunks como indexados
                   await self.pg_store.mark_chunks_indexed(chunk_ids)
                   
                   processed = len(vectors)
                   logger.info(f"Full synchronization successful: {processed} embeddings")
                   
               except Exception as e:
                   logger.error(f"Error during full FAISS synchronization: {e}")
                   failed += len(vectors)
       
       except Exception as e:
           logger.error(f"Error during full synchronization: {e}")
           raise
       finally:
           self.is_syncing = False
       
       return processed, failed
   
   async def verify_sync_status(self) -> Dict[str, Any]:
       """
       Verifica el estado de sincronización entre PostgreSQL y FAISS.
       
       Returns:
           Dict: Estado de sincronización
       """
       faiss_manager = await self.get_faiss_manager()
       
       pg_count = await self.pg_store.get_total_embeddings_count()
       faiss_count = faiss_manager.get_index_info()['total_vectors']
       
       return {
           'postgres_embeddings': pg_count,
           'faiss_vectors': faiss_count,
           'is_synced': pg_count == faiss_count,
           'sync_percentage': (faiss_count / pg_count * 100) if pg_count > 0 else 100,
           'last_check': datetime.now().isoformat(),
           'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
           'stats': self.sync_stats
       }
   
   async def start_scheduler(self, interval_seconds: int = 300):
       """
       Inicia la tarea de sincronización programada.
       
       Args:
           interval_seconds: Intervalo entre sincronizaciones en segundos
       """
       if self.is_scheduler_running:
           logger.warning("Scheduler is already running")
           return
       
       self.is_scheduler_running = True
       
       async def scheduler_task():
           while self.is_scheduler_running:
               try:
                   logger.info(f"Running scheduled synchronization (interval: {interval_seconds}s)")
                   await self.synchronize()
                   
                   # Verificar estado después de sincronización
                   status = await self.verify_sync_status()
                   logger.info(f"Sync status: {status['sync_percentage']:.2f}% synchronized")
                   
               except Exception as e:
                   logger.error(f"Error in scheduler task: {e}")
               
               await asyncio.sleep(interval_seconds)
       
       self._sync_task = asyncio.create_task(scheduler_task())
       logger.info(f"Synchronization scheduler started with interval {interval_seconds}s")
   
   async def stop_scheduler(self):
       """
       Detiene la tarea de sincronización programada.
       """
       if not self.is_scheduler_running:
           return
       
       self.is_scheduler_running = False
       
       if self._sync_task:
           self._sync_task.cancel()
           try:
               await self._sync_task
           except asyncio.CancelledError:
               pass
           self._sync_task = None
       
       logger.info("Synchronization scheduler stopped")
   
   def get_system_status(self) -> Dict[str, Any]:
       """
       Obtiene el estado completo del sistema de sincronización.
       
       Returns:
           Dict: Estado completo del sistema
       """
       faiss_info = None
       if self.faiss_manager:
           faiss_info = self.faiss_manager.get_index_info()
       
       return {
           'is_syncing': self.is_syncing,
           'scheduler_running': self.is_scheduler_running,
           'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
           'stats': self.sync_stats,
           'faiss_index': faiss_info
       }


# Singleton para acceso global
_sync_service = None

async def get_sync_service() -> SyncService:
   """
   Obtiene o crea el servicio de sincronización singleton.
   
   Returns:
       SyncService: Instancia única del servicio
   """
   global _sync_service
   
   if _sync_service is None:
       _sync_service = SyncService()
   
   return _sync_service