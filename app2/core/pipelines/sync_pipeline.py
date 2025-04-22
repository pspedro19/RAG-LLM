#!/usr/bin/env python3
"""
FAISS synchronization pipeline for the RAG system.

This module handles the synchronization between PostgreSQL and FAISS,
ensuring that all embeddings are properly indexed for vector search.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional

from app2.core.config.config import Config
from app2.embeddings.embedding_repository import EmbeddingRepository
from app2.core.faiss_manager import FAISSVectorStore

# Configure logging
logger = logging.getLogger(__name__)

class FAISSsyncPipeline:
    """Pipeline for synchronizing PostgreSQL embeddings with FAISS."""
    
    def __init__(self, config: Optional[Config] = None, use_localhost: bool = True):
        """
        Initialize the FAISS synchronization pipeline.
        
        Args:
            config: Configuration object (optional)
            use_localhost: If True, use localhost instead of postgres_pgvector
        """
        self.config = config or Config()
        
        # Use localhost for the database if specified
        if use_localhost:
            logger.info("Using localhost to connect to PostgreSQL")
            self.config.DB_HOST = "localhost"
        
        # Create components
        self.embedding_repository = EmbeddingRepository(self.config)
        self.faiss_store = FAISSVectorStore(self.config)
        
        logger.info("FAISS synchronization pipeline initialized")
    
    async def close(self):
        """Close connections and clean up resources."""
        await self.embedding_repository.close()
        logger.info("FAISS synchronization pipeline resources closed")
    
    async def synchronize(self) -> Dict[str, Any]:
        """
        Synchronize embeddings from PostgreSQL to FAISS.
        
        Returns:
            Dict with synchronization status
        """
        logger.info("Starting FAISS synchronization...")
        start_time = time.time()
        
        # Import the synchronization function
        from app2.core.sync.sync_faiss import full_synchronization
        
        # Run synchronization
        result = await full_synchronization(batch_size=1000)
        
        duration = time.time() - start_time
        logger.info(f"Synchronization completed in {duration:.2f}s")
        logger.info(f"Status: {result.get('final_status', {}).get('sync_percentage', 0):.2f}% synchronized")
        
        return {
            "postgres_embeddings": result.get('final_status', {}).get('postgres_embeddings', 0),
            "faiss_vectors": result.get('final_status', {}).get('faiss_vectors', 0),
            "sync_percentage": result.get('final_status', {}).get('sync_percentage', 0),
            "duration": duration,
            "batches_processed": result.get('total_batches', 0),
            "embeddings_processed": result.get('total_processed', 0)
        }
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild the FAISS index from scratch.
        
        Returns:
            Dict with rebuild status
        """
        logger.info("Starting FAISS index rebuild...")
        start_time = time.time()
        
        # Import the rebuild function
        from app2.core.sync.sync_faiss import rebuild_faiss_index
        
        # Run rebuild
        result = await rebuild_faiss_index()
        
        duration = time.time() - start_time
        logger.info(f"Index rebuild completed in {duration:.2f}s")
        logger.info(f"Status: {result.get('sync_percentage', 0):.2f}% synchronized")
        
        return {
            "postgres_embeddings": result.get('postgresql_count', 0),
            "faiss_vectors": result.get('faiss_count', 0),
            "sync_percentage": result.get('sync_percentage', 0),
            "duration": duration
        }
    
    async def verify_sync_status(self) -> Dict[str, Any]:
        """
        Verify the synchronization status between PostgreSQL and FAISS.
        
        Returns:
            Dict with synchronization status
        """
        # Get PostgreSQL count
        pg_count = await self.embedding_repository.get_total_embeddings_count()
        
        # Get FAISS count
        faiss_count = self.faiss_store.get_index_info()['total_vectors']
        
        # Calculate sync percentage
        sync_percentage = 100.0 if pg_count == 0 else min(100.0, (faiss_count / pg_count) * 100.0)
        
        logger.info(f"Sync verification: PostgreSQL={pg_count}, FAISS={faiss_count}, {sync_percentage:.2f}%")
        
        return {
            "postgres_embeddings": pg_count,
            "faiss_vectors": faiss_count,
            "sync_percentage": sync_percentage,
            "is_synced": abs(pg_count - faiss_count) <= 5  # Allow small discrepancy
        }

async def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FAISS Synchronization Pipeline')
    parser.add_argument('--rebuild', action='store_true',
                      help='Rebuild the FAISS index from scratch')
    parser.add_argument('--verify', action='store_true',
                      help='Only verify the synchronization status')
    parser.add_argument('--use-container-name', action='store_true',
                      help='Use postgres_pgvector instead of localhost')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )