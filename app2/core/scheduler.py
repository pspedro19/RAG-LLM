# app/core/scheduler.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import signal
import sys

# Importaciones de nuestro propio proyecto
from app.core.config.config import Config
from app.core.sync_service import SyncService

logger = logging.getLogger(__name__)

class Scheduler:
   """
   Scheduler para ejecutar tareas periódicas de sincronización entre
   PostgreSQL (pgvector) y FAISS. Maneja la programación, ejecución,
   y monitoreo del proceso de sincronización.
   """
   
   def __init__(self, sync_service: SyncService, interval_seconds: int = 300):
       """
       Inicializa el scheduler con el servicio de sincronización.
       
       Args:
           sync_service: Servicio que maneja la sincronización entre PostgreSQL y FAISS.
           interval_seconds: Intervalo entre sincronizaciones (default: 300s / 5min).
       """
       self.sync_service = sync_service
       self.interval = interval_seconds
       self.is_running = False
       self.last_sync: Optional[datetime] = None
       self._task: Optional[asyncio.Task] = None
       
       # Registrar handlers para señales de cierre
       self._register_signal_handlers()
   
   def _register_signal_handlers(self):
       """Registra handlers para señales de sistema como SIGINT, SIGTERM."""
       for sig in (signal.SIGINT, signal.SIGTERM):
           signal.signal(sig, self._handle_shutdown_signal)
   
   def _handle_shutdown_signal(self, signum, frame):
       """Handler para señales de cierre del sistema."""
       logger.info(f"Received shutdown signal {signum}, stopping scheduler...")
       asyncio.create_task(self.stop())
       
   async def start(self):
       """
       Inicia el proceso de sincronización automática.
       Crea una tarea asíncrona para el loop de sincronización.
       """
       if self.is_running:
           logger.warning("Scheduler is already running")
           return

       self.is_running = True
       self._task = asyncio.create_task(self._sync_loop())
       logger.info(f"Scheduler started with {self.interval}s interval")

   async def stop(self):
       """
       Detiene el proceso de sincronización automática de manera segura.
       Cancela la tarea de sincronización si está en ejecución.
       """
       if not self.is_running:
           logger.info("Scheduler already stopped")
           return

       logger.info("Stopping scheduler...")
       self.is_running = False
       
       if self._task:
           self._task.cancel()
           try:
               await self._task
           except asyncio.CancelledError:
               pass
       
       logger.info("Scheduler stopped successfully")

   async def _sync_loop(self):
       """
       Loop principal de sincronización.
       Ejecuta la sincronización a intervalos regulares y monitorea el estado.
       """
       while self.is_running:
           try:
               # Ejecutar sincronización
               logger.info(f"Starting scheduled synchronization at {datetime.now().isoformat()}")
               processed, failed = await self.sync_service.synchronize()
               self.last_sync = datetime.now()
               
               logger.info(
                   f"Sync completed - Processed: {processed}, Failed: {failed}, "
                   f"Time: {self.last_sync.isoformat()}"
               )

               # Verificar estado de sincronización
               status = await self.sync_service.verify_sync_status()
               if not status.get('is_synced', False):
                   logger.warning(
                       f"Sync verification failed: PostgreSQL ({status.get('postgres_embeddings', 0)}) "
                       f"and FAISS ({status.get('faiss_vectors', 0)}) are not in sync"
                   )
               else:
                   logger.info(f"Systems in sync with {status.get('postgres_embeddings', 0)} embeddings")

           except Exception as e:
               logger.error(f"Error in sync loop: {e}", exc_info=True)

           # Calcular próxima ejecución
           next_sync = datetime.now() + timedelta(seconds=self.interval)
           logger.info(f"Next sync scheduled for: {next_sync.isoformat()}")
           
           # Esperar hasta el próximo intervalo
           await asyncio.sleep(self.interval)

   def get_status(self) -> dict:
       """
       Retorna el estado actual del scheduler.
       
       Returns:
           dict: Estado del scheduler incluyendo:
               - is_running: Si el scheduler está ejecutándose
               - last_sync: Timestamp de la última sincronización
               - interval_seconds: Intervalo de sincronización configurado
               - next_sync: Timestamp estimado de la próxima sincronización
       """
       next_sync = None
       if self.last_sync and self.is_running:
           next_sync = (self.last_sync + timedelta(seconds=self.interval)).isoformat()
           
       return {
           'is_running': self.is_running,
           'last_sync': self.last_sync.isoformat() if self.last_sync else None,
           'interval_seconds': self.interval,
           'next_sync': next_sync
       }
   
   async def force_sync(self) -> dict:
       """
       Fuerza una sincronización inmediata, independientemente del intervalo programado.
       
       Returns:
           dict: Resultados de la sincronización
       """
       try:
           logger.info("Forcing manual synchronization")
           processed, failed = await self.sync_service.synchronize()
           self.last_sync = datetime.now()
           
           return {
               'status': 'success',
               'processed': processed,
               'failed': failed,
               'timestamp': self.last_sync.isoformat()
           }
       except Exception as e:
           logger.error(f"Error during forced sync: {e}", exc_info=True)
           return {
               'status': 'error',
               'error': str(e),
               'timestamp': datetime.now().isoformat()
           }


# Función para usar este módulo como script independiente
async def run_scheduler(config_path: str = None):
   """
   Ejecuta el scheduler como un proceso independiente.
   
   Args:
       config_path: Ruta opcional al archivo de configuración
   """
   from app.core.config.config import load_config
   from app.core.sync_service import create_sync_service
   
   # Cargar configuración
   config = load_config(config_path) if config_path else Config()
   
   # Crear servicio de sincronización
   sync_service = await create_sync_service(config)
   
   # Crear y ejecutar scheduler
   scheduler = Scheduler(sync_service, interval_seconds=config.SYNC_INTERVAL)
   
   try:
       await scheduler.start()
       # Mantener el proceso vivo
       while True:
           await asyncio.sleep(3600)  # Esperar 1 hora y verificar de nuevo
   except asyncio.CancelledError:
       await scheduler.stop()
   finally:
       # Asegurar que el scheduler se detiene adecuadamente
       if scheduler.is_running:
           await scheduler.stop()


if __name__ == "__main__":
   # Configurar logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   # Ejecutar como script independiente
   asyncio.run(run_scheduler())