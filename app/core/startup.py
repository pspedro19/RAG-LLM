# app/core/startup/startup.py
import asyncio
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI

from app.core.config.config import get_settings
from app.core.vector_store.faiss_manager import get_faiss_manager
from app.core.sync.sync_service import get_sync_service
from app.core.db.service import get_rag_db_service
from app.core.embeddings.service import get_embedding_service

logger = logging.getLogger(__name__)

class RAGStartupService:
   """
   Servicio de inicialización del sistema RAG.
   Orquesta el arranque de los componentes necesarios de forma secuencial.
   """
   
   def __init__(self, app: Optional[FastAPI] = None):
       self.app = app
       self.is_initialized = False
       
   async def initialize(self):
       """
       Inicializa todos los componentes del sistema RAG en el orden correcto.
       """
       if self.is_initialized:
           logger.info("RAG system already initialized")
           return
           
       try:
           logger.info("Initializing RAG system components...")
           
           # 1. Cargar configuración
           settings = get_settings()
           logger.info("Configuration loaded")
           
           # 2. Inicializar servicio de base de datos
           db_service = await get_rag_db_service()
           await self._verify_database_requirements(db_service)
           logger.info("Database service initialized")
           
           # 3. Inicializar servicio de embeddings
           embedding_service = await get_embedding_service()
           logger.info("Embedding service initialized")
           
           # 4. Inicializar gestor de FAISS
           faiss_manager = await get_faiss_manager()
           logger.info("FAISS manager initialized")
           
           # 5. Inicializar servicio de sincronización
           sync_service = await get_sync_service(faiss_manager, db_service)
           
           # 6. Verificar estado de sincronización inicial
           sync_status = await sync_service.verify_sync_status()
           logger.info(f"Initial sync status: {sync_status}")
           
           # 7. Iniciar sincronización automática si está habilitada
           if settings.AUTO_SYNC:
               await sync_service.start_scheduler(interval_seconds=settings.SYNC_INTERVAL)
               logger.info(f"Sync scheduler started with interval: {settings.SYNC_INTERVAL} seconds")
           
           # 8. Registrar eventos de cierre si hay una aplicación FastAPI
           if self.app:
               self._register_shutdown_handlers()
           
           self.is_initialized = True
           logger.info("RAG system initialized successfully")
           
       except Exception as e:
           logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
           raise
   
   async def _verify_database_requirements(self, db_service):
       """
       Verifica que la base de datos cumpla con los requisitos del sistema.
       Delegado a una función específica para mantener el método initialize() limpio.
       """
       # Obtener estadísticas del sistema
       stats = await db_service.get_stats()
       logger.info(f"Database stats: {stats}")
       
       # Verificar extensiones necesarias (delegado al servicio de BD)
       extensions = await db_service.db_service.fetch_all(
           "SELECT extname FROM pg_extension"
       )
       ext_names = [ext['extname'] for ext in extensions]
       
       if 'vector' not in ext_names:
           logger.warning("pgvector extension not found in database")
       else:
           logger.info("pgvector extension is installed")
   
   def _register_shutdown_handlers(self):
       """
       Registra manejadores de cierre para asegurar que los recursos se liberan correctamente.
       """
       @self.app.on_event("shutdown")
       async def shutdown_event():
           logger.info("Shutting down RAG system...")
           
           # Delegar el cierre a los servicios correspondientes
           sync_service = await get_sync_service()
           await sync_service.stop_scheduler()
           
           faiss_manager = await get_faiss_manager()
           faiss_manager.save_index()
           
           db_service = await get_rag_db_service()
           await db_service.close()
           
           logger.info("RAG system shutdown complete")
   
   async def force_synchronize(self) -> Dict[str, Any]:
       """
       Fuerza una sincronización completa entre PostgreSQL y FAISS.
       """
       if not self.is_initialized:
           await self.initialize()
       
       logger.info("Starting forced synchronization...")
       sync_service = await get_sync_service()
       result = await sync_service.synchronize_all()
       
       return result
   
   async def get_system_status(self) -> Dict[str, Any]:
       """
       Obtiene el estado completo del sistema RAG.
       """
       if not self.is_initialized:
           return {"status": "not_initialized"}
       
       try:
           settings = get_settings()
           sync_service = await get_sync_service()
           faiss_manager = await get_faiss_manager()
           db_service = await get_rag_db_service()
           
           sync_status = await sync_service.verify_sync_status()
           faiss_info = faiss_manager.get_index_info()
           db_stats = await db_service.get_stats()
           
           return {
               "status": "initialized",
               "sync_status": sync_status,
               "faiss_info": faiss_info,
               "database_stats": db_stats,
               "auto_sync": sync_service.get_scheduler_status(),
               "config": {
                   "chunk_size": settings.CHUNK_SIZE,
                   "chunk_overlap": settings.CHUNK_OVERLAP,
                   "vector_size": settings.VECTOR_SIZE
               }
           }
       except Exception as e:
           logger.error(f"Error getting system status: {e}")
           return {
               "status": "error",
               "error": str(e)
           }


# Singleton instance for global access
_rag_service = None

async def get_rag_service(app: Optional[FastAPI] = None) -> RAGStartupService:
   """
   Obtiene o crea una instancia singleton del servicio RAG.
   """
   global _rag_service
   
   if _rag_service is None:
       _rag_service = RAGStartupService(app)
       await _rag_service.initialize()
   
   return _rag_service