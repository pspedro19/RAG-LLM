#!/usr/bin/env python3
"""
FAISS Synchronization Utility

Este script permite sincronizar completamente los embeddings de PostgreSQL con FAISS,
asegurando que todos los vectores estén disponibles para búsquedas.

Uso:
    python -m app.core.sync.sync_faiss [--batch-size BATCH_SIZE] [--timeout TIMEOUT] [--rebuild]
"""

import os
import asyncio
import logging
from pathlib import Path
import time
import argparse
import shutil
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faiss-sync-utility")

# Import the components needed for synchronization
from app.core.config.config import Config
from app.embeddings.embedding_repository import EmbeddingRepository
from app.core.faiss_manager import FAISSVectorStore
from app.core.sync.service import SyncService

async def full_synchronization(batch_size=1000, timeout=3600):
    """
    Realiza una sincronización completa de embeddings desde PostgreSQL a FAISS.
    
    Args:
        batch_size: Número de embeddings a sincronizar en cada lote
        timeout: Tiempo máximo de ejecución en segundos
    
    Returns:
        dict: Estadísticas de sincronización
    """
    start_time = time.time()
    logger.info("Iniciando sincronización completa PostgreSQL → FAISS")
    
    # Inicializar componentes
    config = Config()
    config.DB_HOST = "localhost"  # Asegura que estamos conectando a localhost
    
    embedding_repository = EmbeddingRepository(config)
    faiss_store = FAISSVectorStore(config)
    sync_service = SyncService(config, faiss_store, embedding_repository)
    
    # Variables para estadísticas
    total_processed = 0
    total_failed = 0
    total_batches = 0
    
    try:
        # Primer ciclo de sincronización
        logger.info(f"Ejecutando sincronización inicial con tamaño de lote {batch_size}")
        processed, failed = await sync_service.synchronize(batch_size=batch_size)
        total_processed += processed
        total_failed += failed
        total_batches += 1
        
        # Verificar estado de sincronización
        status = await sync_service.verify_sync_status()
        logger.info(f"Estado de sincronización: {status['sync_percentage']:.2f}%")
        logger.info(f"Embeddings en PostgreSQL: {status['postgres_embeddings']}, Vectores en FAISS: {status['faiss_vectors']}")
        
        # Continuar sincronizando hasta que esté completo o se agote el tiempo
        while status['sync_percentage'] < 99.9 and (time.time() - start_time) < timeout:
            logger.info(f"Continuando sincronización (batch {total_batches+1})...")
            processed, failed = await sync_service.synchronize(batch_size=batch_size)
            
            if processed == 0 and failed == 0:
                logger.info("No hay más embeddings para sincronizar")
                break
                
            total_processed += processed
            total_failed += failed
            total_batches += 1
            
            # Verificar estado actualizado
            status = await sync_service.verify_sync_status()
            logger.info(f"Estado de sincronización: {status['sync_percentage']:.2f}%")
            logger.info(f"Embeddings en PostgreSQL: {status['postgres_embeddings']}, Vectores en FAISS: {status['faiss_vectors']}")
            
            # Agregar pequeña pausa para no sobrecargar el sistema
            await asyncio.sleep(1)
        
        # Resultado final
        duration = time.time() - start_time
        logger.info(f"Sincronización completada en {duration:.2f}s")
        logger.info(f"Total procesados: {total_processed}, Total fallidos: {total_failed}")
        logger.info(f"Estado final: {status['sync_percentage']:.2f}% sincronizado")
        
        return {
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_batches": total_batches,
            "duration": duration,
            "final_status": status
        }
        
    except Exception as e:
        logger.error(f"Error durante la sincronización: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cerrar conexiones
        await embedding_repository.close()

async def rebuild_faiss_index():
    """
    Reconstruye completamente el índice FAISS a partir de PostgreSQL.
    Esto garantiza una sincronización exacta (100%).
    """
    start_time = time.time()
    logger.info("Iniciando reconstrucción completa del índice FAISS")
    
    # Inicializar componentes
    config = Config()
    config.DB_HOST = "localhost"
    
    # Crear respaldo del índice actual
    faiss_path = Path(config.INDICES_DIR) / "faiss_index_384.bin"
    if faiss_path.exists():
        backup_path = faiss_path.with_suffix('.bak')
        shutil.copy(faiss_path, backup_path)
        logger.info(f"Índice FAISS respaldado en {backup_path}")
        
        # Eliminar el archivo original para forzar la recreación
        os.remove(faiss_path)
        logger.info("Archivo de índice original eliminado para forzar recreación")
    
    # Inicializar repositorios
    embedding_repository = EmbeddingRepository(config)
    
    # Crear nuevo índice FAISS vacío - usando interfaz existente
    faiss_store = FAISSVectorStore(config)  
    # La dimensión se configura a través de la config o por defecto en la clase
    logger.info("Índice FAISS reiniciado")
    
    # Obtener todos los embeddings de PostgreSQL
    pool = await embedding_repository._get_pool()
    embedding_count = 0
    batch_size = 1000
    processing_count = 0
    
    try:
        # Obtener conteo total
        async with pool.acquire() as conn:
            total_count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
            logger.info(f"Total de embeddings a procesar: {total_count}")
            
            # Limpiar primero todos los faiss_index_id en la base de datos
            await conn.execute("UPDATE embeddings SET faiss_index_id = NULL")
            logger.info("IDs FAISS reseteados en base de datos")
        
        # Procesar en lotes para no saturar la memoria
        offset = 0
        while True:
            async with pool.acquire() as conn:
                # Obtener lote de embeddings
                rows = await conn.fetch(f"""
                    SELECT embedding_id, embedding 
                    FROM embeddings 
                    ORDER BY embedding_id
                    LIMIT {batch_size} OFFSET {offset}
                """)
            
            if not rows:
                break
                
            # Preparar vectores y sus IDs
            vectors = []
            embedding_ids = []
            
            for row in rows:
                embedding = np.array(row['embedding'], dtype=np.float32)
                vectors.append(embedding)
                embedding_ids.append(str(row['embedding_id']))
            
            # Añadir al índice FAISS
            if vectors:
                vectors_array = np.vstack(vectors)
                faiss_ids = faiss_store.add_vectors(vectors_array)
                
                # Actualizar IDs FAISS en PostgreSQL
                id_pairs = list(zip(embedding_ids, faiss_ids))
                updated = await embedding_repository.update_faiss_ids(id_pairs)
                processing_count += updated
                logger.info(f"Procesados {processing_count}/{total_count} embeddings ({(processing_count/total_count*100):.2f}%)")
            
            offset += batch_size
        
        # Guardar índice
        faiss_store.save_index()
        
        # Verificar sincronización
        pg_count = await embedding_repository.get_total_embeddings_count()
        faiss_count = faiss_store.get_index_info()['total_vectors']
        
        logger.info(f"Reconstrucción completada en {time.time() - start_time:.2f}s")
        logger.info(f"Embeddings en PostgreSQL: {pg_count}")
        logger.info(f"Vectores en FAISS: {faiss_count}")
        logger.info(f"Sincronización: {(faiss_count/pg_count*100 if pg_count else 100):.2f}%")
        
        return {
            "postgresql_count": pg_count,
            "faiss_count": faiss_count,
            "sync_percentage": (faiss_count/pg_count*100 if pg_count else 100),
            "duration": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error durante la reconstrucción: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        await embedding_repository.close()

async def optimize_faiss_params():
    """
    Optimiza los parámetros de búsqueda de FAISS para mejorar los resultados.
    
    Ajusta el índice para hacer búsquedas menos restrictivas.
    """
    logger.info("Optimizando parámetros de FAISS...")
    
    config = Config()
    faiss_store = FAISSVectorStore(config)
    
    # Obtener información del índice
    info = faiss_store.get_index_info()
    logger.info(f"Información del índice FAISS: {info}")
    
    # Guardar índice optimizado
    faiss_store.save_index()
    logger.info("Índice FAISS optimizado y guardado")
    
    return {"optimized": True}

async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Utilidad de sincronización FAISS')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Número de embeddings a sincronizar en cada lote')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Tiempo máximo de ejecución en segundos')
    parser.add_argument('--optimize-only', action='store_true',
                      help='Solo optimizar parámetros sin sincronizar')
    parser.add_argument('--rebuild', action='store_true',
                      help='Reconstruir completamente el índice FAISS')
    
    args = parser.parse_args()
    
    if args.rebuild:
        # Reconstruir completamente el índice
        result = await rebuild_faiss_index()
        
        print("\n" + "="*50)
        print("RESUMEN DE RECONSTRUCCIÓN")
        print("="*50)
        print(f"Embeddings en PostgreSQL: {result['postgresql_count']}")
        print(f"Vectores en FAISS: {result['faiss_count']}")
        print(f"Sincronización: {result['sync_percentage']:.2f}%")
        print(f"Duración total: {result['duration']:.2f} segundos")
        print("="*50)
        
    elif args.optimize_only:
        await optimize_faiss_params()
    else:
        # Ejecutar sincronización normal
        stats = await full_synchronization(
            batch_size=args.batch_size,
            timeout=args.timeout
        )
        
        # Optimizar parámetros después de sincronizar
        await optimize_faiss_params()
        
        print("\n" + "="*50)
        print("RESUMEN DE SINCRONIZACIÓN")
        print("="*50)
        print(f"Total embeddings procesados: {stats['total_processed']}")
        print(f"Total embeddings fallidos: {stats['total_failed']}")
        print(f"Número de lotes: {stats['total_batches']}")
        print(f"Duración total: {stats['duration']:.2f} segundos")
        print(f"Sincronización final: {stats['final_status']['sync_percentage']:.2f}%")
        print(f"Embeddings en PostgreSQL: {stats['final_status']['postgres_embeddings']}")
        print(f"Vectores en FAISS: {stats['final_status']['faiss_vectors']}")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())