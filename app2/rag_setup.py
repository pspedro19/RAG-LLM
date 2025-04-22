#!/usr/bin/env python3
"""
RAG Setup Script - Herramienta para configurar el sistema RAG

Este script permite:
1. Cargar documentos desde un directorio
2. Reconstruir índices FAISS
3. Verificar el estado del sistema RAG

Uso:
    python rag_setup.py --help
"""

import asyncio
import argparse
import os
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Añadir directorio padre al path para importar módulos
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-setup")

# Importar componentes necesarios
from app2.core.config.config import Config
from app2.core.pipelines.ingest_pipeline import DocumentIngestionPipeline
from app2.core.pipelines.sync_pipeline import FAISSsyncPipeline
from app2.embeddings.embedding_repository import EmbeddingRepository
from app2.core.faiss_manager import FAISSVectorStore

class RAGSetup:
    """Clase para configurar y gestionar el sistema RAG."""
    
    def __init__(self, use_localhost: bool = True):
        """
        Inicializa el configurador del sistema RAG.
        
        Args:
            use_localhost: Si es True, usa localhost en lugar de postgres_pgvector
        """
        self.config = Config()
        
        # Usar localhost para la base de datos si se especifica
        if use_localhost:
            logger.info("Usando localhost para conectar a PostgreSQL")
            self.config.DB_HOST = "localhost"
        
        # Inicializar componentes
        self.ingest_pipeline = None
        self.sync_pipeline = None
        self.embedding_repository = EmbeddingRepository(self.config)
        self.faiss_store = FAISSVectorStore(self.config)
        
        logger.info("RAG Setup inicializado")
    
    async def close(self):
        """Cierra conexiones y limpia recursos."""
        if self.ingest_pipeline:
            await self.ingest_pipeline.close()
        if self.sync_pipeline:
            await self.sync_pipeline.close()
        await self.embedding_repository.close()
        logger.info("Recursos de RAG Setup cerrados")
    
    async def ingest_documents(self, doc_dir: str, limit: Optional[int] = None) -> List[str]:
        """
        Ingiere documentos de un directorio.
        
        Args:
            doc_dir: Directorio con documentos a procesar
            limit: Número máximo de documentos a procesar
            
        Returns:
            Lista de IDs de documentos procesados
        """
        if not os.path.exists(doc_dir):
            logger.error(f"Directorio no encontrado: {doc_dir}")
            return []
            
        logger.info(f"Ingiriendo documentos desde: {doc_dir}")
        
        # Inicializar pipeline si es necesario
        if not self.ingest_pipeline:
            self.ingest_pipeline = DocumentIngestionPipeline(config=self.config)
        
        # Procesar documentos
        doc_ids = await self.ingest_pipeline.process_documents(doc_dir, limit=limit)
        
        logger.info(f"Documentos procesados: {len(doc_ids)}")
        return doc_ids
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """
        Reconstruye el índice FAISS desde cero.
        
        Returns:
            Diccionario con estado de la reconstrucción
        """
        logger.info("Reconstruyendo índice FAISS...")
        
        # Inicializar pipeline si es necesario
        if not self.sync_pipeline:
            self.sync_pipeline = FAISSsyncPipeline(config=self.config)
        
        # Reconstruir índice
        result = await self.sync_pipeline.rebuild_index()
        
        logger.info(f"Reconstrucción completada: {result['sync_percentage']:.2f}% sincronizado")
        return result
    
    async def synchronize_index(self) -> Dict[str, Any]:
        """
        Sincroniza el índice FAISS con los embeddings en PostgreSQL.
        
        Returns:
            Diccionario con estado de la sincronización
        """
        logger.info("Sincronizando índice FAISS...")
        
        # Inicializar pipeline si es necesario
        if not self.sync_pipeline:
            self.sync_pipeline = FAISSsyncPipeline(config=self.config)
        
        # Sincronizar índice
        result = await self.sync_pipeline.synchronize()
        
        logger.info(f"Sincronización completada: {result['sync_percentage']:.2f}% sincronizado")
        return result
    
    async def verify_status(self) -> Dict[str, Any]:
        """
        Verifica el estado del sistema RAG.
        
        Returns:
            Diccionario con estado del sistema
        """
        logger.info("Verificando estado del sistema RAG...")
        
        # Inicializar pipeline si es necesario
        if not self.sync_pipeline:
            self.sync_pipeline = FAISSsyncPipeline(config=self.config)
        
        # Verificar estado de sincronización
        sync_status = await self.sync_pipeline.verify_sync_status()
        
        # Obtener estadísticas de embeddings
        emb_stats = await self.embedding_repository.get_stats()
        
        # Obtener estadísticas de FAISS
        faiss_info = self.faiss_store.get_index_info()
        
        # Combinar toda la información
        status = {
            "sync_status": sync_status,
            "embeddings_stats": emb_stats,
            "faiss_info": faiss_info,
            "config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "vector_size": self.config.VECTOR_SIZE
            }
        }
        
        logger.info(f"Estado: {sync_status['sync_percentage']:.2f}% sincronizado")
        logger.info(f"Embeddings en PostgreSQL: {sync_status['postgres_embeddings']}")
        logger.info(f"Vectores en FAISS: {sync_status['faiss_vectors']}")
        
        return status

def print_status_summary(status):
    """Imprime un resumen del estado del sistema."""
    print("\n" + "=" * 60)
    print("RESUMEN DEL SISTEMA RAG")
    print("=" * 60)
    
    sync_status = status.get("sync_status", {})
    emb_stats = status.get("embeddings_stats", {})
    faiss_info = status.get("faiss_info", {})
    
    print(f"Embeddings en PostgreSQL: {sync_status.get('postgres_embeddings', 0)}")
    print(f"Vectores en FAISS: {sync_status.get('faiss_vectors', 0)}")
    print(f"Estado de sincronización: {sync_status.get('sync_percentage', 0):.2f}%")
    print(f"¿Completamente sincronizado? {'Sí' if sync_status.get('is_synced', False) else 'No'}")
    
    print("\nDistribución de modelos:")
    for model, count in emb_stats.get("models", {}).items():
        print(f"  - {model}: {count} embeddings")
        
    print("\nInformación del índice FAISS:")
    print(f"  - Tipo: {faiss_info.get('index_type', 'Desconocido')}")
    print(f"  - Dimensión: {faiss_info.get('dimension', 0)}")
    print(f"  - Total vectores: {faiss_info.get('total_vectors', 0)}")
    
    print("=" * 60)

async def main():
    """Función principal para ejecución desde la línea de comandos."""
    parser = argparse.ArgumentParser(description='RAG Setup - Herramienta para configurar el sistema RAG')
    
    # Subparsers para comandos
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando ingerir
    ingest_parser = subparsers.add_parser('ingest', help='Ingerir documentos')
    ingest_parser.add_argument('--doc-dir', type=str, required=True, 
                              help='Directorio con documentos a procesar')
    ingest_parser.add_argument('--limit', type=int, default=None,
                              help='Número máximo de documentos a procesar')
    ingest_parser.add_argument('--use-container-name', action='store_true',
                              help='Usar postgres_pgvector en lugar de localhost')
    
    # Comando reconstruir
    rebuild_parser = subparsers.add_parser('rebuild', help='Reconstruir índice FAISS')
    rebuild_parser.add_argument('--use-container-name', action='store_true',
                              help='Usar postgres_pgvector en lugar de localhost')
    
    # Comando sincronizar
    sync_parser = subparsers.add_parser('sync', help='Sincronizar índice FAISS')
    sync_parser.add_argument('--use-container-name', action='store_true',
                            help='Usar postgres_pgvector en lugar de localhost')
    
    # Comando verificar
    status_parser = subparsers.add_parser('status', help='Verificar estado del sistema RAG')
    status_parser.add_argument('--use-container-name', action='store_true',
                            help='Usar postgres_pgvector en lugar de localhost')
    
    # Comando wizard (interactivo)
    wizard_parser = subparsers.add_parser('wizard', help='Asistente interactivo para configurar el sistema')
    wizard_parser.add_argument('--use-container-name', action='store_true',
                            help='Usar postgres_pgvector en lugar de localhost')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    use_localhost = not getattr(args, 'use_container_name', False)
    
    # Inicializar RAG Setup
    setup = RAGSetup(use_localhost=use_localhost)
    
    try:
        if args.command == 'ingest':
            # Ingerir documentos
            await setup.ingest_documents(args.doc_dir, args.limit)
            
            # Sincronizar índice automáticamente después de ingerir
            print("\nSincronizando índice FAISS con los nuevos documentos...")
            await setup.synchronize_index()
            
        elif args.command == 'rebuild':
            # Reconstruir índice
            result = await setup.rebuild_index()
            print(f"\nÍndice reconstruido: {result['sync_percentage']:.2f}% sincronizado")
            
        elif args.command == 'sync':
            # Sincronizar índice
            result = await setup.synchronize_index()
            print(f"\nÍndice sincronizado: {result['sync_percentage']:.2f}% sincronizado")
            
        elif args.command == 'status':
            # Verificar estado
            status = await setup.verify_status()
            print_status_summary(status)
            
        elif args.command == 'wizard':
            # Asistente interactivo
            await run_wizard(setup)
            
    finally:
        # Cerrar recursos
        await setup.close()

async def run_wizard(setup: RAGSetup):
    """Ejecuta un asistente interactivo para configurar el sistema RAG."""
    print("\n" + "=" * 60)
    print("ASISTENTE DE CONFIGURACIÓN DEL SISTEMA RAG")
    print("=" * 60)
    
    while True:
        print("\nSeleccione una opción:")
        print("1. Ingerir documentos")
        print("2. Reconstruir índice FAISS")
        print("3. Sincronizar índice FAISS")
        print("4. Verificar estado del sistema")
        print("0. Salir")
        
        option = input("\nOpción: ")
        
        if option == '0':
            print("Saliendo del asistente...")
            break
            
        elif option == '1':
            # Ingerir documentos
            doc_dir = input("Directorio de documentos: ")
            if not os.path.exists(doc_dir):
                print(f"Error: El directorio '{doc_dir}' no existe.")
                continue
                
            limit_input = input("Límite de documentos (Enter para no limitar): ")
            limit = int(limit_input) if limit_input.strip() else None
            
            await setup.ingest_documents(doc_dir, limit)
            
            sync_option = input("¿Desea sincronizar el índice FAISS? (s/n): ")
            if sync_option.lower() in ('s', 'si', 'sí', 'y', 'yes'):
                await setup.synchronize_index()
                
        elif option == '2':
            # Reconstruir índice
            confirm = input("¿Está seguro de que desea reconstruir el índice FAISS? (s/n): ")
            if confirm.lower() in ('s', 'si', 'sí', 'y', 'yes'):
                result = await setup.rebuild_index()
                print(f"\nÍndice reconstruido: {result['sync_percentage']:.2f}% sincronizado")
                
        elif option == '3':
            # Sincronizar índice
            result = await setup.synchronize_index()
            print(f"\nÍndice sincronizado: {result['sync_percentage']:.2f}% sincronizado")
            
        elif option == '4':
            # Verificar estado
            status = await setup.verify_status()
            print_status_summary(status)
            
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)