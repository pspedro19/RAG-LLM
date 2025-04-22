#!/usr/bin/env python3
"""
Document ingestion pipeline for the RAG system.

This module handles the processing of documents, chunking, and embedding generation.
"""

import logging
import asyncio
from pathlib import Path
import time
import traceback
from typing import List, Dict, Any, Optional

from app2.core.config.config import Config
from app2.ingestion.document_reader import DocumentReader
from app2.ingestion.document_processor import DocumentProcessor
from app2.ingestion.chunk_repository import ChunkRepository
from app2.embeddings.embedding_service import EmbeddingService
from app2.embeddings.embedding_repository import EmbeddingRepository

# Configure logging
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """Pipeline for document ingestion and embedding generation."""
    
    def __init__(self, config: Optional[Config] = None, use_localhost: bool = True):
        """
        Initialize the document ingestion pipeline.
        
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
        self.document_reader = DocumentReader()
        self.document_processor = DocumentProcessor(self.config)
        self.chunk_repository = ChunkRepository(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.embedding_repository = EmbeddingRepository(self.config)
        
        # Store information about processed documents
        self.doc_ids = []
        self.chunk_counts = {}
        self.embedding_counts = {}
        
        logger.info("Document ingestion pipeline initialized")
    
    async def close(self):
        """Close connections and clean up resources."""
        await self.chunk_repository.close()
        await self.embedding_repository.close()
        logger.info("Document ingestion pipeline resources closed")
    
    async def process_documents(self, doc_dir: str, limit: Optional[int] = None) -> List[str]:
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

async def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Ingestion Pipeline')
    parser.add_argument('--doc-dir', type=str, required=True,
                      help='Directory with documents to process')
    parser.add_argument('--limit', type=int, default=None,
                      help='Maximum number of documents to process')
    parser.add_argument('--use-container-name', action='store_true',
                      help='Use postgres_pgvector instead of localhost')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = DocumentIngestionPipeline(use_localhost=not args.use_container_name)
    try:
        await pipeline.process_documents(args.doc_dir, limit=args.limit)
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())