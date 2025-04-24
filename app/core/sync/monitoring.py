# app/core/sync/monitoring.py
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config.config import Config

logger = logging.getLogger(__name__)

class SyncMonitor:
    """
    Monitor para el servicio de sincronización entre PostgreSQL y FAISS.
    
    Se encarga de programar sincronizaciones periódicas y monitorear su estado.
    Proporciona métricas y estadísticas sobre el proceso de sincronización.
    """
    
    def __init__(self, sync_service, config: Optional[Config] = None):
        """
        Inicializa el monitor de sincronización.
        
        Args:
            sync_service: Servicio de sincronización a monitorear
            config: Configuración del sistema
        """
        self.sync_service = sync_service
        self.config = config or Config()
        
        # State tracking
        self._monitor_task = None
        self.is_monitoring = False
        
        # Métricas y estadísticas
        self.metrics = {
            'total_syncs': 0,
            'total_checks': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'last_successful_sync': None,
            'last_check_time': None,
            'avg_sync_duration': 0.0,
            'is_monitoring': False
        }
    
    async def check_sync_status(self) -> Dict[str, Any]:
        """
        Verifica el estado actual de sincronización.
        
        Returns:
            Dict con información sobre el estado de sincronización
        """
        try:
            self.metrics['total_checks'] += 1
            self.metrics['last_check_time'] = datetime.now().isoformat()
            
            return await self.sync_service.verify_sync_status()
        except Exception as e:
            logger.error(f"Error checking synchronization status: {e}")
            return {
                'error': str(e),
                'last_check': self.last_check_time.isoformat(),
                'is_synced': False
            }
    
    async def perform_sync_with_metrics(self) -> Dict[str, Any]:
        """
        Realiza una sincronización y registra métricas sobre el proceso.
        
        Returns:
            Dict con resultados y métricas de la sincronización
        """
        start_time = time.time()
        result = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'processed': 0,
            'failed': 0,
            'duration': 0.0
        }
        
        try:
            processed, failed = await self.sync_service.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            
            # Actualizar métricas globales
            self.metrics['total_syncs'] += 1
            
            if failed == 0:
                self.metrics['successful_syncs'] += 1
                self.metrics['last_successful_sync'] = datetime.now().isoformat()
                result['status'] = 'success'
            else:
                self.metrics['failed_syncs'] += 1
                result['status'] = 'partial' if processed > 0 else 'failed'
            
            # Actualizar duración promedio
            total_success = self.metrics['successful_syncs']
            if total_success > 0:
                current_avg = self.metrics['avg_sync_duration']
                self.metrics['avg_sync_duration'] = ((current_avg * (total_success - 1)) + duration) / total_success
            
            # Llenar el resultado
            result['processed'] = processed
            result['failed'] = failed
            result['duration'] = duration
            
            return result
        except Exception as e:
            self.metrics['sync_failures'] += 1
            logger.error(f"Error during synchronization: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - start_time
            }
    
    async def start_monitoring(self, check_interval: int = None):
        """
        Inicia el monitoreo periódico del estado de sincronización.
        
        Args:
            check_interval: Intervalo entre comprobaciones en segundos
        """
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        if check_interval is None:
            check_interval = self.config.SYNC_INTERVAL
        
        self.is_monitoring = True
        self.metrics['is_monitoring'] = True
        
        async def monitor_task():
            while self.is_monitoring:
                try:
                    logger.info(f"Running sync check (interval: {check_interval}s)")
                    
                    # Verificar estado de sincronización
                    status = await self.check_sync_status()
                    
                    # Si no está sincronizado, iniciar sincronización
                    if not status.get('is_synced', False) and self.config.AUTO_SYNC:
                        logger.info("Detected desynchronization, starting sync")
                        sync_result = await self.perform_sync_with_metrics()
                        logger.info(f"Sync result: {sync_result['status']}")
                    
                except Exception as e:
                    logger.error(f"Error in monitoring task: {e}")
                
                await asyncio.sleep(check_interval)
        
        self._monitor_task = asyncio.create_task(monitor_task())
        logger.info(f"Sync monitoring started with interval {check_interval}s")
    
    async def stop_monitoring(self):
        """
        Detiene el monitoreo periódico.
        """
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.metrics['is_monitoring'] = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        logger.info("Sync monitoring stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas actuales del monitor.
        
        Returns:
            Dict con métricas y estadísticas
        """
        # Actualizar estado de monitoreo
        self.metrics['is_monitoring'] = self.is_monitoring
        return self.metrics
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """
        Obtiene un estado detallado del sistema de sincronización.
        
        Returns:
            Dict con estado detallado y métricas
        """
        sync_status = await self.check_sync_status()
        system_status = self.sync_service.get_system_status()
        
        return {
            'sync_status': sync_status,
            'system_status': system_status,
            'metrics': self.get_metrics(),
            'faiss_info': system_status.get('faiss_index', None),
            'timestamp': datetime.now().isoformat()
        }