#!/usr/bin/env python3
"""
Command-line interface for the RAG system operations.

This module provides commands for document ingestion, FAISS synchronization,
and search functionality.

Usage:
    python -m app2.pipelines.cli ingest --doc-dir /path/to/documents
    python -m app2.pipelines.cli sync --rebuild
    python -m app2.pipelines.cli search --query "Your search query"
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from app2.core.pipelines.ingest_pipeline import DocumentIngestionPipeline
from app2.core.pipelines.sync_pipeline import FAISSsyncPipeline
from app2.core.pipelines.search_pipeline import SearchPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-system-cli")

async def run_ingest(doc_dir, limit=None, use_localhost=True):
    """Run the document ingestion pipeline."""
    pipeline = DocumentIngestionPipeline(use_localhost=use_localhost)
    try:
        doc_ids = await pipeline.process_documents(doc_dir, limit=limit)
        print(f"\n{'='*50}")
        print("INGESTION SUMMARY")
        print(f"{'='*50}")
        print(f"Documents processed: {len(doc_ids)}")
        print(f"Total chunks: {sum(pipeline.chunk_counts.values())}")
        print(f"Total embeddings: {sum(pipeline.embedding_counts.values())}")
        print(f"{'='*50}")
        return doc_ids
    finally:
        await pipeline.close()

async def run_sync(rebuild=False, use_localhost=True):
    """Run the FAISS synchronization pipeline."""
    pipeline = FAISSsyncPipeline(use_localhost=use_localhost)
    try:
        if rebuild:
            result = await pipeline.rebuild_index()
        else:
            result = await pipeline.synchronize()
        
        print(f"\n{'='*50}")
        print("SYNCHRONIZATION SUMMARY")
        print(f"{'='*50}")
        print(f"Embeddings in PostgreSQL: {result['postgres_embeddings']}")
        print(f"Vectors in FAISS: {result['faiss_vectors']}")
        print(f"Synchronization: {result['sync_percentage']:.2f}%")
        print(f"{'='*50}")
        
        return result
    finally:
        await pipeline.close()

async def run_search(query, mode="hybrid", top_k=5, strategy="relevance", use_localhost=True, full_content=False, no_cross_lingual=False):
    """Run the search pipeline."""
    config_kwargs = {}
    if no_cross_lingual:
        config_kwargs['ENABLE_CROSS_LINGUAL'] = False
        
    pipeline = SearchPipeline(use_localhost=use_localhost, config=None, **config_kwargs)
    try:
        result = await pipeline.search(query, mode=mode, top_k=top_k, strategy=strategy)
        
        print(f"\n{'='*50}")
        print("SEARCH RESULTS")
        print(f"{'='*50}")
        print(f"Query: '{query}'")
        
        # Mostrar consulta traducida si está disponible
        if 'query' in result and 'translated_text' in result['query']:
            print(f"Translated query: '{result['query']['translated_text']}'")
            if 'languages' in result.get('search_metadata', {}):
                langs = result['search_metadata']['languages']
                print(f"Languages: {langs[0]} → {langs[1]}")
        
        print(f"Mode: {mode}")
        print(f"Strategy: {strategy}")
        print(f"Results found: {len(result.get('chunks', []))}")
        
        if full_content:
            # Mostrar contenido completo de cada chunk
            print("\nFULL CONTENT:")
            print(f"{'='*50}")
            for i, chunk in enumerate(result.get('chunks', [])):
                lang = f"[{chunk.get('source_language', 'unknown').upper()}]"
                doc_title = chunk.get('title', 'Unknown')
                content = chunk.get('content', '')
                similarity = chunk.get('similarity', None)
                info_score = chunk.get('info_score', None)
                
                print(f"\n{'-'*80}")
                print(f"CHUNK {i+1}: {lang} [Documento: {doc_title}]")
                if similarity is not None:
                    print(f"Similarity: {similarity:.4f}", end="")
                if info_score is not None:
                    print(f", Info score: {info_score:.4f}", end="")
                print("\n" + "-"*80)
                print(content)
            print(f"{'='*50}")
        else:
            # Mostrar contexto en formato tradicional
            if result.get('context'):
                print("\nContext:")
                print(f"{'='*50}")
                print(result['context'])
                print(f"{'='*50}")
            else:
                print("\nNo context generated from search results")
        
        return result
    finally:
        await pipeline.close()

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='RAG System Operations')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('--doc-dir', type=str, required=True, 
                              help='Directory with documents to process')
    ingest_parser.add_argument('--limit', type=int, default=None,
                              help='Maximum number of documents to process')
    ingest_parser.add_argument('--use-container-name', action='store_true',
                              help='Use postgres_pgvector instead of localhost')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Synchronize FAISS index')
    sync_parser.add_argument('--rebuild', action='store_true',
                            help='Rebuild the FAISS index from scratch')
    sync_parser.add_argument('--use-container-name', action='store_true',
                            help='Use postgres_pgvector instead of localhost')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for information')
    search_parser.add_argument('--query', type=str, required=True,
                              help='Search query')
    search_parser.add_argument('--mode', choices=['faiss', 'pgvector', 'hybrid'],
                              default='hybrid', help='Search mode')
    search_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of results to return')
    search_parser.add_argument('--strategy', choices=['relevance', 'chronological', 'by_document'],
                              default='relevance', help='Strategy for context building')
    search_parser.add_argument('--use-container-name', action='store_true',
                              help='Use postgres_pgvector instead of localhost')
    search_parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], 
                              default='info', help='Logging level')
    search_parser.add_argument('--full-content', action='store_true',
                              help='Display full content of each chunk')
    search_parser.add_argument('--no-cross-lingual', action='store_true',
                              help='Disable cross-lingual search')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    use_localhost = not getattr(args, 'use_container_name', False)
    
    try:
        if args.command == 'ingest':
            asyncio.run(run_ingest(args.doc_dir, args.limit, use_localhost))
        elif args.command == 'sync':
            asyncio.run(run_sync(args.rebuild, use_localhost))
        elif args.command == 'search':
            # Configurar nivel de logging
            if hasattr(args, 'log_level'):
                log_level = getattr(logging, args.log_level.upper())
                logging.getLogger().setLevel(log_level)
                # También ajustar loggers específicos de nuestra app
                for logger_name in ['app2', 'app2.query', 'app2.core.pipelines']:
                    logging.getLogger(logger_name).setLevel(log_level)
                    
            # Ejecutar búsqueda con parámetros adicionales
            full_content = getattr(args, 'full_content', False)
            no_cross_lingual = getattr(args, 'no_cross_lingual', False)
            
            asyncio.run(run_search(
                args.query, 
                args.mode, 
                args.top_k, 
                args.strategy, 
                use_localhost,
                full_content=full_content,
                no_cross_lingual=no_cross_lingual
            ))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()