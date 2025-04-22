#!/usr/bin/env python3
"""
Integration test with localhost connection for the RAG system.

Modified version of the integration test that uses localhost to
connect to PostgreSQL when running outside Docker.

Usage:
    python test_integration_local.py
"""

import os
import asyncio
import logging
from pathlib import Path
import time
import argparse
import tempfile
import shutil
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-integration-test")

# Import the RAG system components
from app2.core.config.config import Config
from app2.ingestion.document_reader import DocumentReader
from app2.ingestion.document_processor import DocumentProcessor
from app2.ingestion.chunk_repository import ChunkRepository
from app2.embeddings.embedding_service import EmbeddingService
from app2.embeddings.embedding_repository import EmbeddingRepository
from app2.core.faiss_manager import FAISSVectorStore
from app2.core.sync.service import SyncService
from app2.query.query_processor import QueryProcessor
from app2.query.retriever import Retriever
from app2.query.context_builder import ContextBuilder

class RAGIntegrationTest:
    """Integration test for the RAG system."""
    
    def __init__(self, use_temp_dir=False, use_localhost=True):
        """
        Initialize the integration test.
        
        Args:
            use_temp_dir: If True, use a temporary directory for FAISS indices
            use_localhost: If True, use localhost instead of postgres_pgvector
        """
        self.temp_dir = None
        if use_temp_dir:
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Using temporary directory for FAISS indices: {self.temp_dir}")
        
        # Initialize configuration
        self.config = Config()
        if self.temp_dir:
            self.config.INDICES_DIR = Path(self.temp_dir)
        
        # Use localhost for the database if specified
        if use_localhost:
            logger.info("Using localhost to connect to PostgreSQL")
            self.config.DB_HOST = "localhost"
        
        # Create components
        self.document_reader = DocumentReader()
        self.document_processor = DocumentProcessor(self.config)
        self.chunk_repository = ChunkRepository(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.embedding_repository = EmbeddingRepository(self.config)
        self.faiss_store = FAISSVectorStore(self.config)
        self.sync_service = SyncService(self.config, self.faiss_store, self.embedding_repository)
        self.query_processor = QueryProcessor(self.config, self.embedding_service)
        self.retriever = Retriever(self.faiss_store, self.embedding_repository, self.config)
        self.context_builder = ContextBuilder(self.config, self.chunk_repository)
        
        # Store information about processed documents
        self.doc_ids = []
        self.chunk_counts = {}
        self.embedding_counts = {}
        
        logger.info("RAG system initialized")
    
    async def close(self):
        """Close connections and clean up resources."""
        await self.chunk_repository.close()
        await self.embedding_repository.close()
        
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Temporary directory removed: {self.temp_dir}")
    
    async def process_documents(self, doc_dir, limit=None):
        """
        Process documents in a directory and store them in the system.
        
        Args:
            doc_dir: Directory containing the documents to process
            limit: Maximum number of documents to process (None = no limit)
            
        Returns:
            List of processed document IDs
        """
        logger.info(f"Processing documents from: {doc_dir}")
        
        # Get list of documents
        documents = []
        pdf_dir = Path(doc_dir)
        
        # Look for PDFs in subdirectories
        for subdir in pdf_dir.glob("*"):
            if subdir.is_dir():
                for pdf_file in subdir.glob("*.pdf"):
                    documents.append((pdf_file, subdir.name))
        
        if not documents:
            logger.warning(f"No PDF documents found in {doc_dir}")
            return []
        
        logger.info(f"Found {len(documents)} PDF documents")
        if limit:
            documents = documents[:limit]
            logger.info(f"Processing the first {limit} documents")
        
        total_chunks = 0
        total_embeddings = 0
        
        # Process each document
        for doc_path, category in documents:
            start_time = time.time()
            logger.info(f"Processing: {doc_path.name} (Category: {category})")
            
            try:
                # Read document
                doc_data = self.document_reader.read_document(str(doc_path))
                
                # Verify data type and extract content
                doc_content = ""
                if isinstance(doc_data, dict):
                    if isinstance(doc_data.get('content'), str):
                        doc_content = doc_data.get('content', '')
                    elif isinstance(doc_data.get('content'), dict):
                        # If content is a dictionary, try to convert it to string
                        try:
                            import json
                            doc_content = json.dumps(doc_data.get('content', {}))
                        except Exception as e:
                            logger.error(f"Error serializing content: {e}")
                            doc_content = str(doc_data.get('content', {}))
                    else:
                        doc_content = str(doc_data.get('content', ''))
                    logger.debug(f"Document read as dictionary, extracting content")
                else:
                    doc_content = str(doc_data) if doc_data is not None else ""
                    logger.debug(f"Document read as: {type(doc_data)}")
                
                # Create document in DB
                doc_id = await self.chunk_repository.create_document(
                    title=doc_path.name,
                    metadata={"source": "pdf", "category": category}
                )
                self.doc_ids.append(doc_id)
                
                # Process document
                document_input = {
                    'content': doc_content,  # Use the extracted content
                    'metadata': {"source": "pdf", "category": category},
                    'doc_id': doc_id,
                    'title': doc_path.name
                }
                
                # Verify content before processing
                if not document_input['content']:
                    logger.warning(f"Empty content for {doc_path.name}, skipping processing")
                    continue
                
                # Try to extract text directly from PDF if needed
                if len(document_input['content']) < 100:
                    logger.warning(f"Content seems too short, attempting direct PDF extraction")
                    try:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(str(doc_path))
                        text_content = ""
                        for page in reader.pages:
                            text_content += page.extract_text() + "\n\n"
                        
                        if len(text_content) > len(document_input['content']):
                            document_input['content'] = text_content
                            logger.info(f"Extracted {len(text_content)} characters directly from PDF")
                    except Exception as e:
                        logger.error(f"Error extracting text directly from PDF: {e}")
                
                # Adapt to document_processor expectations
                try:
                    chunks = self.document_processor.create_chunks(document_input)
                except TypeError as e:
                    if "expected string" in str(e):
                        logger.warning("Attempting to adapt format for document_processor...")
                        # If it expects a string, we pass only the text
                        chunks = self.document_processor.create_chunks(document_input['content'])
                    else:
                        raise
                
                # Add necessary metadata
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, str):
                        # If chunks are strings, convert to dictionaries
                        chunks[i] = {
                            'content': chunk,
                            'doc_id': doc_id,
                            'chunk_number': i
                        }
                    else:
                        # If dictionaries, ensure they have the required fields
                        if 'doc_id' not in chunk:
                            chunk['doc_id'] = doc_id
                        if 'chunk_number' not in chunk:
                            chunk['chunk_number'] = i
                
                # Store chunks
                chunk_ids = await self.chunk_repository.store_chunks(doc_id, chunks)
                self.chunk_counts[doc_id] = len(chunk_ids)
                total_chunks += len(chunk_ids)
                
                # Generate and store embeddings
                texts = [chunk['content'] for chunk in chunks]
                embeddings = self.embedding_service.generate_embeddings(texts)
                
                embedding_ids = []
                for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                    embedding_id = await self.embedding_repository.store_embedding(
                        chunk_id=chunk_id,
                        embedding=embedding,
                        model_name='miniLM'
                    )
                    embedding_ids.append(embedding_id)
                
                self.embedding_counts[doc_id] = len(embedding_ids)
                total_embeddings += len(embedding_ids)
                
                duration = time.time() - start_time
                logger.info(f"✓ Document processed in {duration:.2f}s: {len(chunks)} chunks, {len(embedding_ids)} embeddings")
                
            except Exception as e:
                logger.error(f"Error processing {doc_path.name}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Processing completed: {len(self.doc_ids)} documents, {total_chunks} chunks, {total_embeddings} embeddings")
        return self.doc_ids
    
    async def synchronize_faiss(self):
        """Synchronize embeddings from PostgreSQL with FAISS using full rebuild approach."""
        logger.info("Synchronizing embeddings with FAISS (full rebuild)...")
        start_time = time.time()
        
        # Import the rebuild function from your sync script
        from app2.core.sync.sync_faiss import rebuild_faiss_index
        
        # Run full rebuild to ensure 100% synchronization
        result = await rebuild_faiss_index()
        
        duration = time.time() - start_time
        logger.info(f"Synchronization completed in {duration:.2f}s")
        logger.info(f"Synchronization status: {result['sync_percentage']:.2f}% synchronized")
        logger.info(f"Embeddings in PostgreSQL: {result['postgresql_count']}, Vectors in FAISS: {result['faiss_count']}")
        
        return {
            "postgres_embeddings": result["postgresql_count"],
            "faiss_vectors": result["faiss_count"],
            "sync_percentage": result["sync_percentage"]
        }
    
    async def verify_sync_status(self):
        """
        Modified verification of synchronization status between PostgreSQL and FAISS.
        Replaces the call to sync_service.verify_sync_status() that caused the error.
        
        Returns:
            dict: Synchronization status
        """
        # Get PostgreSQL count with a direct query
        pg_count = 0
        try:
            query = "SELECT COUNT(*) FROM embeddings"
            # Use connection from chunk_repository
            result = await self.chunk_repository.conn.fetchrow(query)
            pg_count = result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting embedding count from PostgreSQL: {e}")
            try:
                # Alternative approach
                stats = await self.embedding_repository.get_stats()
                pg_count = stats.get('total_embeddings', 0)
            except Exception as e2:
                logger.error(f"Failed alternative approach: {e2}")
            
        # Get FAISS count
        faiss_count = 0
        try:
            # Try different methods to get FAISS count
            if hasattr(self.faiss_store, 'index') and hasattr(self.faiss_store.index, 'ntotal'):
                faiss_count = self.faiss_store.index.ntotal
            elif hasattr(self.faiss_store, 'get_index_info'):
                info = self.faiss_store.get_index_info()
                if isinstance(info, dict) and 'total_vectors' in info:
                    faiss_count = info['total_vectors']
        except Exception as e:
            logger.error(f"Error getting vector count from FAISS: {e}")
        
        # Calculate sync percentage
        sync_percentage = 100.0 if pg_count == 0 else min(100.0, (faiss_count / pg_count) * 100.0)
        
        return {
            "postgres_embeddings": pg_count,
            "faiss_vectors": faiss_count,
            "sync_percentage": sync_percentage
        }
    
    async def perform_search(self, query_text, search_mode='faiss', top_k=5):
        """
        Perform a search in the RAG system.
        
        Args:
            query_text: Query text
            search_mode: Search mode ('faiss', 'pgvector', 'hybrid')
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        logger.info(f"Performing search: '{query_text}' (mode: {search_mode}, top_k: {top_k})")
        start_time = time.time()
        
        # Process query
        query_data = await self.query_processor.process_query(query_text)
        logger.info(f"Query processed - Language: {query_data['language']}, Type: {query_data['query_type']}")
        
        # Check if there are embeddings in FAISS for search
        faiss_vector_count = 0
        try:
            if hasattr(self.faiss_store, 'index') and hasattr(self.faiss_store.index, 'ntotal'):
                faiss_vector_count = self.faiss_store.index.ntotal
        except Exception:
            pass
            
        if faiss_vector_count == 0 and search_mode in ['faiss', 'hybrid']:
            logger.warning(f"No vectors in FAISS to search. Mode: {search_mode}")
            return {'chunks': [], 'context': f"No information available to answer the query: '{query_text}'"}
        
        # Perform search with vector
        try:
            # Realizar la búsqueda
            retrieval_results = await self.retriever.retrieve(
                query_vector=query_data['embedding'],
                k=top_k,
                mode=search_mode
            )
            
            # Construir contexto a partir de los chunk_ids encontrados
            if 'chunk_ids' in retrieval_results and retrieval_results.get('chunk_ids'):
                context_result = await self.context_builder.build_context(
                    chunk_ids=retrieval_results['chunk_ids'],
                    similarities=retrieval_results.get('similarities', [[]])[0] if retrieval_results.get('similarities') else None,
                    strategy='relevance',
                    max_chunks=top_k
                )
                
                # Añadir la clave 'chunks' para compatibilidad con el código existente
                retrieval_results['chunks'] = context_result.get('chunks', [])
            else:
                # Si no hay resultados en la búsqueda vectorial, añadir lista vacía
                retrieval_results['chunks'] = []
            
            # Si no hay resultados, intentar búsqueda por palabras clave como fallback
            if not retrieval_results['chunks'] and hasattr(self.embedding_repository, 'search_by_keywords'):
                logger.info(f"No vector search results, trying keyword search for: {query_text}")
                keyword_results = await self.embedding_repository.search_by_keywords(
                    keywords=query_text,
                    limit=top_k
                )
                
                if keyword_results:
                    # Obtener chunk_ids de los resultados de palabras clave
                    chunk_ids = [r.get('chunk_id') for r in keyword_results if r.get('chunk_id')]
                    if chunk_ids:
                        # Construir contexto con estos chunk_ids
                        keyword_context = await self.context_builder.build_context(
                            chunk_ids=chunk_ids,
                            strategy='relevance',
                            max_chunks=top_k
                        )
                        retrieval_results['chunks'] = keyword_context.get('chunks', [])
                        retrieval_results['context'] = keyword_context.get('context', '')
                    else:
                        # Si no hay chunk_ids válidos, usar directamente los resultados
                        retrieval_results['chunks'] = [
                            {
                                'chunk_id': result.get('chunk_id'),
                                'content': result.get('content', '')[:100] + "...",
                                'similarity': result.get('similarity', 0.5),
                                'title': result.get('title', 'Unknown')
                            }
                            for result in keyword_results
                        ]
                        
                    logger.info(f"Found {len(retrieval_results['chunks'])} results via keyword search")
            
            # Build context
            context_result = await self.context_builder.build_from_retrieval_results(
                retrieval_results=retrieval_results,
                strategy='relevance'
            )
            
            # Asegurar que context_result tiene la clave 'chunks'
            if 'chunks' not in context_result and 'chunks' in retrieval_results:
                context_result['chunks'] = retrieval_results['chunks']
            elif 'chunks' not in context_result:
                context_result['chunks'] = []
            
            duration = time.time() - start_time
            logger.info(f"Search completed in {duration:.2f}s: {len(context_result['chunks'])} chunks found")
            
            # Show results
            if context_result.get('context'):
                logger.info(f"Generated context ({len(context_result['context'])} characters):")
                logger.info("------------------------------------------")
                logger.info(context_result['context'][:500] + "..." if len(context_result['context']) > 500 else context_result['context'])
                logger.info("------------------------------------------")
            else:
                logger.info("No context generated from search results")
            
            return context_result
        except Exception as e:
            logger.error(f"Error in {search_mode} search: {e}")
            logger.error(traceback.format_exc())
            return {'chunks': [], 'context': f"Error searching for information: '{query_text}'"}
    
    async def run_integration_test(self, doc_dir, queries, doc_limit=None):
        """
        Run the complete integration test.
        
        Args:
            doc_dir: Directory with documents to process
            queries: List of queries to test
            doc_limit: Document processing limit
            
        Returns:
            Test results
        """
        results = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'sync_status': None,
            'search_results': []
        }
        
        try:
            # 1. Process documents
            doc_ids = await self.process_documents(doc_dir, limit=doc_limit)
            results['documents_processed'] = len(doc_ids)
            results['total_chunks'] = sum(self.chunk_counts.values())
            results['total_embeddings'] = sum(self.embedding_counts.values())
            
            # 2. Synchronize with FAISS
            sync_status = await self.synchronize_faiss()
            results['sync_status'] = sync_status
            
            # 3. Perform searches
            for query in queries:
                # Try FAISS search
                faiss_results = await self.perform_search(query, search_mode='faiss')
                results['search_results'].append({
                    'query': query,
                    'mode': 'faiss',
                    'chunks_found': len(faiss_results.get('chunks', [])),
                    'context_length': len(faiss_results.get('context', ''))
                })
                
                # Try pgvector search if available
                try:
                    pgvector_results = await self.perform_search(query, search_mode='pgvector')
                    results['search_results'].append({
                        'query': query,
                        'mode': 'pgvector',
                        'chunks_found': len(pgvector_results.get('chunks', [])),
                        'context_length': len(pgvector_results.get('context', ''))
                    })
                except Exception as e:
                    logger.warning(f"pgvector search not available: {e}")
                
                # Try hybrid search if available
                try:
                    hybrid_results = await self.perform_search(query, search_mode='hybrid')
                    results['search_results'].append({
                        'query': query,
                        'mode': 'hybrid',
                        'chunks_found': len(hybrid_results.get('chunks', [])),
                        'context_length': len(hybrid_results.get('context', ''))
                    })
                except Exception as e:
                    logger.warning(f"Hybrid search not available: {e}")
            
            logger.info("Integration test completed successfully")
            return results
        
        except Exception as e:
            logger.error(f"Error in integration test: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Optional cleanup
            # If doc_limit is very low, we can clean up the documents to avoid filling the DB
            if doc_limit and doc_limit <= 3:
                for doc_id in self.doc_ids:
                    try:
                        await self.chunk_repository.delete_document(doc_id)
                    except Exception as e:
                        logger.warning(f"Error deleting document {doc_id}: {e}")


async def main():
    """Main function to run the integration test."""
    parser = argparse.ArgumentParser(description='Integration test for RAG system')
    parser.add_argument('--doc-dir', type=str, default='/root/LLMBitlink/app2/data/documents/pdf/curaçao_information',
                        help='Directory with PDF documents')
    parser.add_argument('--doc-limit', type=int, default=None,
                        help='Maximum number of documents to process (None = process all)')
    parser.add_argument('--temp-dir', action='store_true',
                        help='Use temporary directory for FAISS indices')
    parser.add_argument('--use-container-name', action='store_true',
                        help='Use postgres_pgvector instead of localhost')
    
    args = parser.parse_args()
    
    # English test queries specific to Curaçao documents
    queries = [
        "What are the main tourist attractions in Curaçao?",
        "Information about Curaçao's economy",
        "What is the history of Curaçao?",
        "Tell me about education in Curaçao",
        "What are the best beaches in Curaçao?"
    ]
    
    logger.info("Starting RAG system integration test")
    test = RAGIntegrationTest(
        use_temp_dir=args.temp_dir,
        use_localhost=not args.use_container_name
    )
    
    try:
        results = await test.run_integration_test(args.doc_dir, queries, doc_limit=args.doc_limit)
        
        # Show results summary
        print("\n" + "="*50)
        print("INTEGRATION TEST SUMMARY")
        print("="*50)
        print(f"Documents processed: {results['documents_processed']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Total embeddings: {results['total_embeddings']}")
        
        if results['sync_status']:
            print(f"Synchronization: {results['sync_status']['sync_percentage']:.2f}%")
        
        print("\nSearch results:")
        for i, result in enumerate(results['search_results']):
            print(f"  {i+1}. Query: '{result['query']}'")
            print(f"     Mode: {result['mode']}")
            print(f"     Chunks found: {result['chunks_found']}")
            print(f"     Context length: {result['context_length']} characters")
            print()
        
    finally:
        await test.close()


if __name__ == "__main__":
    asyncio.run(main())