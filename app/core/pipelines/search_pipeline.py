#!/usr/bin/env python3
"""
Search pipeline for the RAG system.

This module handles the search functionality, including query processing,
retrieval, and context building with improved filtering, ranking, and diversity.
Includes cross-lingual search capabilities.
"""

import logging
import asyncio
import time
import traceback
import re
from typing import Dict, Any, Optional, List, Tuple

from app.core.config.config import Config
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.embedding_repository import EmbeddingRepository
from app.core.faiss_manager import FAISSVectorStore
from app.query.query_processor import QueryProcessor
from app.query.retriever import Retriever
from app.query.context_builder import ContextBuilder
from app.ingestion.chunk_repository import ChunkRepository

# Configure logging
logger = logging.getLogger(__name__)

def is_toc_chunk(chunk_text):
    """
    Check if a chunk appears to be a table of contents.
    Mejorado para mejor discriminación entre TOCs y contenido valioso.
    """
    if not chunk_text:
        return False
        
    if len(chunk_text) > 100:
        logger.debug(f"Pipeline TOC check: {chunk_text[:100]}... (len: {len(chunk_text)})")
    else:
        logger.debug(f"Pipeline TOC check: {chunk_text}")
        
    # Verificar títulos explícitos de TOC
    toc_indicators = [
        "Table of Contents", "Contents", "List of Tables", "List of Figures",
        "Índice", "Contenidos", "Lista de Tablas", "Lista de Figuras"
    ]
    
    # Verificar marcadores de título (coincidencias exactas o al principio de línea)
    for indicator in toc_indicators:
        # Buscar coincidencia exacta como título
        if re.search(r'(^|\n)[ \t]*' + re.escape(indicator) + r'[ \t]*($|\n)', chunk_text, re.IGNORECASE):
            logger.debug(f"Pipeline: TOC por título exacto: '{indicator}'")
            return True
    
    # Verificar si hay muchas entradas "Figure X" o "Table X" seguidas
    figure_table_pattern = re.compile(r'(Figure|Table|Figura|Tabla)\s+\d+')
    matches = figure_table_pattern.findall(chunk_text)
    lines = chunk_text.split('\n')
    
    if len(matches) > 3 and len(matches) / (len(lines) or 1) > 0.4:
        logger.debug(f"Pipeline: Detectado como lista de figuras/tablas ({len(matches)} ocurrencias)")
        return True
    
    # Excepción específica para tablas de datos que no son TOC
    if re.search(r'Table \d+:', chunk_text):
        dots_ratio = chunk_text.count('.') / len(chunk_text) if chunk_text else 0
        if dots_ratio < 0.1:  # Pocos puntos (típico en tablas de datos, no en TOC)
            logger.debug("Pipeline: Contiene 'Table X:' pero parece ser una tabla de datos, no un TOC")
            return False
    
    # Verificar formato tipo TOC
    toc_line_count = 0
    
    for i, line in enumerate(lines[:15]):  # Verificar más líneas
        # Detectar líneas con formato TOC
        if (re.search(r'\d+\.\d+', line) and re.search(r'\.{3,}\s*\d+$', line)) or \
           (re.search(r'^\s*[A-Z][^.]+\s+\.{3,}\s*\d+$', line)):  # "Capítulo ... 10"
            toc_line_count += 1
            logger.debug(f"Pipeline: Línea TOC {i+1}: '{line}'")
    
    toc_line_ratio = toc_line_count / len(lines) if len(lines) > 0 else 0
    logger.debug(f"Pipeline: Ratio líneas TOC: {toc_line_ratio:.3f}")
    
    # Umbral ajustado (40% en lugar de 50%)
    if len(lines) > 5 and toc_line_ratio > 0.4:
        logger.debug(f"Pipeline: TOC por ratio líneas: {toc_line_ratio:.3f} > 0.4")
        return True
        
    # Verificar densidad de puntos y números
    dots_count = chunk_text.count('.')
    numbers_count = sum(c.isdigit() for c in chunk_text)
    ratio = (dots_count + numbers_count) / len(chunk_text) if chunk_text else 0
    
    # Verificar si hay muchos puntos suspensivos (típico en TOC)
    ellipsis_count = chunk_text.count('...')
    if ellipsis_count > 3 and ellipsis_count / len(lines) > 0.2:
        logger.debug(f"Pipeline: Alto número de puntos suspensivos: {ellipsis_count}")
        return True
    
    # Umbral ajustado (22% en lugar de 20%)
    if ratio > 0.22 and '\n' in chunk_text and chunk_text.count('\n') > 8:
        logger.debug(f"Pipeline: TOC por densidad puntos/números: {ratio:.4f} > 0.22")
        return True
    
    logger.debug("Pipeline: No es TOC")
    return False

def calculate_info_score(chunk_text):
    """
    Calcula un puntaje de informatividad para un contenido.
    Los puntajes más altos indican contenido más informativo.
    """
    if not chunk_text or len(chunk_text) < 50:
        return 0.1
        
    # Factores positivos
    positive_factors = []
    
    # 1. Longitud del contenido (más largo suele ser más informativo)
    length_score = min(1.0, len(chunk_text) / 1000)
    positive_factors.append(length_score)
    
    # 2. Presencia de datos numéricos y porcentajes
    percentage_count = len(re.findall(r'\d+%|\d+\.\d+%', chunk_text))
    number_count = sum(c.isdigit() for c in chunk_text)
    number_density = number_count / len(chunk_text) if chunk_text else 0
    
    # Ajustar para penalizar TOCs (que tienen muchos números pero no son informativos)
    if percentage_count > 0:
        number_score = min(1.0, 0.5 + percentage_count / 10)  # Bonificación por porcentajes
    elif 0.05 <= number_density <= 0.15:  # Rango saludable para texto informativo
        number_score = 0.7
    else:
        number_score = max(0.1, min(0.6, number_density * 4))  # Penalización para valores extremos
        
    positive_factors.append(number_score)
    
    # 3. Presencia de patrones informativos
    info_patterns = [
        re.compile(r'\d+%'),  # Porcentajes
        re.compile(r'(?i)key (finding|result)s?:'),  # Hallazgos clave
        re.compile(r'(?i)conclusion'),  # Conclusiones
        re.compile(r'(?i)importantly'),  # Marcadores de importancia
    ]
    info_pattern_matches = sum(1 for p in info_patterns if p.search(chunk_text))
    info_pattern_score = min(1.0, info_pattern_matches / 2)
    positive_factors.append(info_pattern_score)
    
    # 4. Densidad de texto (proporción de palabras vs. caracteres)
    words = re.findall(r'\b\w+\b', chunk_text)
    word_density = len(words) / len(chunk_text) if chunk_text else 0
    word_density_score = min(1.0, word_density * 7)  # Normalizar a escala 0-1
    positive_factors.append(word_density_score)
    
    # Factores negativos
    negative_factors = []
    
    # 1. Muchos puntos suspensivos (típico en TOCs)
    ellipsis_density = chunk_text.count('...') / len(chunk_text) if chunk_text else 0
    ellipsis_penalty = max(0, min(0.8, ellipsis_density * 50))
    negative_factors.append(ellipsis_penalty)
    
    # 2. Líneas cortas repetitivas (típico en índices)
    lines = chunk_text.split('\n')
    short_lines = sum(1 for line in lines if len(line.strip()) < 40)
    short_line_ratio = short_lines / len(lines) if lines else 0
    short_line_penalty = max(0, min(0.7, short_line_ratio))
    negative_factors.append(short_line_penalty)
    
    # Calcular puntaje final
    positive_score = sum(positive_factors) / len(positive_factors)
    negative_score = sum(negative_factors) / len(negative_factors) if negative_factors else 0
    
    # Combinar con peso mayor para positivos
    final_score = positive_score * 0.7 - negative_score * 0.3
    
    # Normalizar a rango 0-1
    final_score = max(0.0, min(1.0, final_score))
    
    logger.debug(f"Score informativo: {final_score:.3f} (positivo: {positive_score:.2f}, negativo: {negative_score:.2f})")
    return final_score

def show_chunk_details(chunk, prefix=""):
    """Muestra detalles de un chunk para diagnóstico."""
    content = chunk.get('content', '')
    chunk_id = chunk.get('chunk_id', 'unknown')
    title = chunk.get('title', 'sin título')
    
    logger.debug(f"{prefix} Chunk ID: {chunk_id}, Título: {title}")
    
    # Estadísticas
    dots_ratio = content.count('.') / len(content) if content else 0
    numbers_ratio = sum(c.isdigit() for c in content) / len(content) if content else 0
    lines = content.split('\n')
    
    logger.debug(f"{prefix} Estadísticas: {len(content)} chars, {len(lines)} líneas")
    logger.debug(f"{prefix} Ratios: puntos={dots_ratio:.4f}, números={numbers_ratio:.4f}")
    logger.debug(f"{prefix} Info score: {calculate_info_score(content):.3f}")
    
    # Muestra inicio del contenido
    if len(content) > 200:
        logger.debug(f"{prefix} Contenido: {content[:200]}...")
    else:
        logger.debug(f"{prefix} Contenido: {content}")

def filter_low_quality_chunks(chunks, min_content_length=50, mode='hybrid'):
    """
    Filter out chunks that are likely not useful (TOC, blank, etc.).
    Mejorado para usar score informativo y preservar diversidad.
    """
    if not chunks:
        return []
        
    filtered_chunks = []
    filter_reasons = {"short": 0, "toc": 0, "low_info": 0, "other": 0}
    
    # Más permisivo para modo FAISS
    if mode == 'faiss':
        min_content_length = 30
        logger.debug(f"Modo {mode}: usando min_length={min_content_length}")
    
    logger.debug(f"Filtrando {len(chunks)} chunks (modo: {mode})")
    
    # Primero calcular scores informativos para todos los chunks
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        # Añadir score informativo si no existe
        if 'info_score' not in chunk:
            chunk['info_score'] = calculate_info_score(content)
    
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        chunk_id = chunk.get('chunk_id', f'unknown-{i}')
        title = chunk.get('title', 'sin título')
        info_score = chunk.get('info_score', 0)
        
        logger.debug(f"Analizando chunk {i+1}/{len(chunks)}: {chunk_id} - {title} (info_score: {info_score:.3f})")
        
        # Filtrar contenido muy corto
        if len(content) < min_content_length:
            filter_reasons["short"] += 1
            logger.debug(f"FILTRADO - Demasiado corto: {len(content)} < {min_content_length}")
            continue
            
        # Filtrar TOC-like content
        if is_toc_chunk(content):
            filter_reasons["toc"] += 1
            logger.debug(f"FILTRADO - Detectado como TOC")
            continue
            
        # Filtrar contenido de baja informatividad
        min_info_threshold = 0.25 if mode == 'hybrid' else 0.2  # Más exigente en modo híbrido
        if info_score < min_info_threshold:
            filter_reasons["low_info"] += 1
            logger.debug(f"FILTRADO - Baja informatividad: {info_score:.3f} < {min_info_threshold}")
            continue
            
        logger.debug(f"ACEPTADO - Chunk pasa los filtros")
        filtered_chunks.append(chunk)
    
    if len(chunks) > len(filtered_chunks):
        logger.info(f"Filtered out {len(chunks) - len(filtered_chunks)} low-quality chunks "
                  f"(short={filter_reasons['short']}, toc={filter_reasons['toc']}, "
                  f"low_info={filter_reasons['low_info']}, other={filter_reasons['other']})")
        
    # Si todos fueron filtrados, recuperar al menos el chunk con mayor score informativo
    if not filtered_chunks and chunks:
        best_chunk = max(chunks, key=lambda x: x.get('info_score', 0))
        logger.warning(f"Todos los chunks fueron filtrados, recuperando el mejor (info_score: {best_chunk.get('info_score', 0):.3f})")
        filtered_chunks = [best_chunk]
        
    return filtered_chunks

def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks based on content.
    Usa un umbral más alto (0.98) para ser menos agresivo.
    """
    if not chunks:
        return []
        
    seen_content = {}
    unique_chunks = []
    
    for chunk in chunks:
        content = chunk.get('content', '')
        
        # Use a hash of content to detect exact duplicates
        content_hash = hash(content)
        
        if content_hash in seen_content:
            logger.debug(f"Removed exact duplicate chunk (hash match)")
            continue
            
        # También verificar similitud muy alta para near-duplicates
        is_duplicate = False
        for seen_hash, seen_content_text in seen_content.items():
            # Solo comparar texto si las longitudes son similares
            if abs(len(content) - len(seen_content_text)) / max(len(content), len(seen_content_text)) > 0.2:
                continue
                
            # Calcular similitud básica
            shorter, longer = (content, seen_content_text) if len(content) < len(seen_content_text) else (seen_content_text, content)
            similarity = len(shorter) / len(longer) if len(longer) > 0 else 0
            
            # Umbral alto (0.98) para considerar duplicado
            if similarity > 0.98:
                is_duplicate = True
                logger.debug(f"Removed near-duplicate chunk (sim={similarity:.3f})")
                break
                
        if not is_duplicate:
            seen_content[content_hash] = content
            unique_chunks.append(chunk)
            
    if len(chunks) > len(unique_chunks):
        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        
    return unique_chunks

def diversify_chunks(chunks, max_per_doc=2, min_chunks=None):
    """
    Diversifica chunks para no sobrerrepresentar un solo documento.
    
    Args:
        chunks: Lista de chunks a diversificar
        max_per_doc: Máximo número de chunks por documento
        min_chunks: Número mínimo de chunks a devolver
        
    Returns:
        Lista diversificada de chunks
    """
    if not chunks or max_per_doc <= 0:
        return chunks
    
    min_chunks = min_chunks or max(3, len(chunks) // 2)
    
    # Agrupar chunks por documento
    chunks_by_doc = {}
    for chunk in chunks:
        doc_id = chunk.get('doc_id', 'unknown')
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    # Si hay un solo documento, devolver los mejores chunks
    if len(chunks_by_doc) <= 1:
        # Ordenar por score informativo
        if 'info_score' in chunks[0]:
            chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
        elif 'similarity' in chunks[0]:
            chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return chunks[:min(len(chunks), min_chunks)]
    
    # Diversificar - primero tomar los mejores de cada documento
    diverse_chunks = []
    
    # Primera ronda: un chunk de cada documento (el mejor)
    for doc_id, doc_chunks in chunks_by_doc.items():
        # Ordenar por relevancia o info_score
        if 'info_score' in doc_chunks[0]:
            doc_chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
        elif 'similarity' in doc_chunks[0]:
            doc_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Añadir el mejor chunk
        diverse_chunks.append(doc_chunks[0])
    
    # Segunda ronda: añadir segundos mejores chunks hasta llenar max_per_doc
    if len(diverse_chunks) < min_chunks:
        for doc_id, doc_chunks in chunks_by_doc.items():
            if len(doc_chunks) > 1 and len([c for c in diverse_chunks if c.get('doc_id') == doc_id]) < max_per_doc:
                for chunk in doc_chunks[1:max_per_doc]:
                    diverse_chunks.append(chunk)
                    # Detenerse si alcanzamos el mínimo
                    if len(diverse_chunks) >= min_chunks:
                        break
                if len(diverse_chunks) >= min_chunks:
                    break
    
    # Ordenar por relevancia o info_score final
    if diverse_chunks and 'info_score' in diverse_chunks[0]:
        diverse_chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
    elif diverse_chunks and 'similarity' in diverse_chunks[0]:
        diverse_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    # Log de diversificación
    if len(diverse_chunks) < len(chunks):
        logger.info(f"Diversified chunks: {len(chunks)} -> {len(diverse_chunks)} (from {len(chunks_by_doc)} documents)")
        
    return diverse_chunks

def combine_and_rank_chunks(chunks: List[Dict], max_results: int = 10) -> List[Dict]:
    """
    Combina, ordena y limita chunks usando una estrategia de ranking que considera
    tanto la similitud como la calidad del contenido.
    
    Args:
        chunks: Lista de chunks a combinar y ordenar
        max_results: Número máximo de resultados a devolver
        
    Returns:
        Lista ordenada y limitada de chunks
    """
    if not chunks:
        return []
    
    # Asegurar que todos los chunks tengan info_score
    for chunk in chunks:
        if 'info_score' not in chunk:
            chunk['info_score'] = calculate_info_score(chunk.get('content', ''))
    
    # Normalizar similitud si existe
    has_similarity = all('similarity' in chunk for chunk in chunks)
    if has_similarity:
        # Encontrar máxima similitud para normalización
        max_sim = max(chunk.get('similarity', 0) for chunk in chunks)
        if max_sim > 0:
            for chunk in chunks:
                chunk['norm_similarity'] = chunk.get('similarity', 0) / max_sim
        else:
            for chunk in chunks:
                chunk['norm_similarity'] = 0
    
    # Calcular score compuesto
    for chunk in chunks:
        # Si hay similitud, combinar con info_score
        if has_similarity:
            # 60% similitud, 40% info_score
            chunk['combined_score'] = 0.6 * chunk.get('norm_similarity', 0) + 0.4 * chunk.get('info_score', 0)
        else:
            # Solo usar info_score
            chunk['combined_score'] = chunk.get('info_score', 0)
    
    # Ordenar por score compuesto
    chunks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    
    # Diversificar y limitar resultados
    diverse_chunks = diversify_chunks(chunks, max_per_doc=max(1, max_results // 3), min_chunks=max_results)
    
    return diverse_chunks[:max_results]

def deduplicate_bilingual_chunks(chunks):
    """
    Elimina chunks duplicados entre idiomas basado en similitud de contenido
    
    Args:
        chunks: Lista combinada de chunks en diferentes idiomas
    
    Returns:
        Lista sin duplicados, priorizando el idioma original de la consulta
    """
    if not chunks:
        return []
        
    unique_chunks = []
    seen_contents = set()
    
    for chunk in chunks:
        content = chunk.get('content', '')
        # Simplificar contenido para comparación (eliminar espacios extra y puntuación)
        simplified = re.sub(r'\s+', ' ', content).lower()
        simplified = re.sub(r'[.,;:?!]', '', simplified)
        
        # Considerar solo primeros 100 caracteres para comparación entre idiomas
        # (permite detectar traducciones del mismo contenido)
        content_key = simplified[:100]
        
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_chunks.append(chunk)
    
    return unique_chunks

class SearchPipeline:
    """Pipeline for search operations in the RAG system."""
    
    def __init__(self, config: Optional[Config] = None, use_localhost: bool = True):
        """
        Initialize the search pipeline.
        
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
        self.chunk_repository = ChunkRepository(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.embedding_repository = EmbeddingRepository(self.config)
        self.faiss_store = FAISSVectorStore(self.config)
        self.query_processor = QueryProcessor(self.config, self.embedding_service)
        self.retriever = Retriever(self.faiss_store, self.embedding_repository, self.config)
        self.context_builder = ContextBuilder(self.config, self.chunk_repository)
        
        logger.info("Search pipeline initialized")
    
    async def close(self):
        """Close connections and clean up resources."""
        await self.chunk_repository.close()
        await self.embedding_repository.close()
        logger.info("Search pipeline resources closed")
    
    async def _execute_search(self, query_data, mode, top_k, query_text, strategy):
        """
        Ejecuta una búsqueda específica con los parámetros dados.
        Esta es una función privada que implementa la lógica principal de búsqueda.
        
        Args:
            query_data: Datos procesados de la consulta
            mode: Modo de búsqueda (faiss, pgvector, hybrid)
            top_k: Número de resultados a devolver
            query_text: Texto original de la consulta
            strategy: Estrategia de ordenamiento
        
        Returns:
            Resultados de la búsqueda
        """
        # Check if there are embeddings in FAISS for search
        faiss_vector_count = 0
        try:
            if hasattr(self.faiss_store, 'index') and hasattr(self.faiss_store.index, 'ntotal'):
                faiss_vector_count = self.faiss_store.index.ntotal
        except Exception:
            pass
            
        if faiss_vector_count == 0 and mode in ['faiss', 'hybrid']:
            logger.warning(f"No vectors in FAISS to search. Mode: {mode}")
            return {'chunks': [], 'context': f"No information available to answer the query: '{query_text}'"}
        
        # Request more initial results for filtering (3× the final count)
        search_k = top_k * 3
        
        # Perform the search
        retrieval_results = await self.retriever.retrieve(
            query_vector=query_data['embedding'],
            k=search_k,
            mode=mode,
            query_text=query_text  # Pass original query text for context-aware scoring
        )
        
        # Build context from the chunk_ids found
        if 'chunk_ids' in retrieval_results and retrieval_results.get('chunk_ids'):
            chunk_ids = retrieval_results['chunk_ids']
            logger.debug(f"Búsqueda inicial encontró {len(chunk_ids)} chunks: {chunk_ids}")
            
            # Mostrar chunks sin procesar para diagnóstico
            raw_chunks = await self.context_builder.get_chunks_by_ids(chunk_ids)
            logger.debug(f"Mostrando {min(3, len(raw_chunks))} de {len(raw_chunks)} chunks originales:")
            for i, chunk in enumerate(raw_chunks[:3]):
                show_chunk_details(chunk, f"[RAW {i+1}]")
            
            # Filtrar chunks de baja calidad
            filtered_chunks = filter_low_quality_chunks(raw_chunks, mode=mode)
            
            # Eliminar duplicados
            unique_chunks = remove_duplicate_chunks(filtered_chunks)
            
            # Reordenar chunk_ids y similitudes basados en chunk filtrados
            filtered_chunk_ids = [chunk.get('chunk_id') for chunk in unique_chunks]
            
            # Recuperar similitudes si existen
            similarities = retrieval_results.get('similarities', [[]])[0] if retrieval_results.get('similarities') else None
            
            # Si tenemos similitudes, mapearlas a los chunks filtrados
            if similarities and len(similarities) == len(chunk_ids):
                filtered_similarities = []
                for chunk_id in filtered_chunk_ids:
                    if chunk_id in chunk_ids:
                        idx = chunk_ids.index(chunk_id)
                        if idx < len(similarities):
                            filtered_similarities.append(similarities[idx])
                        else:
                            filtered_similarities.append(0)
                    else:
                        filtered_similarities.append(0)
            else:
                filtered_similarities = None
            
            # Construir contexto con chunks filtrados
            if filtered_chunk_ids:
                context_result = await self.context_builder.build_context(
                    chunk_ids=filtered_chunk_ids,
                    similarities=filtered_similarities,
                    strategy=strategy,
                    max_chunks=top_k
                )
                
                # Añadir chunks al resultado
                retrieval_results['chunks'] = context_result.get('chunks', [])
                retrieval_results['context'] = context_result.get('context', '')
            else:
                # Si no hay chunks después del filtrado, probar una estrategia más permisiva
                logger.warning("Todos los chunks fueron filtrados, intentando con criterios más permisivos")
                
                # Criterios más permisivos
                raw_chunks = await self.context_builder.get_chunks_by_ids(chunk_ids)
                for chunk in raw_chunks:
                    chunk['info_score'] = calculate_info_score(chunk.get('content', ''))
                
                # Ordenar por info_score y tomar los mejores
                raw_chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
                best_chunks = raw_chunks[:min(top_k, len(raw_chunks))]
                
                if best_chunks:
                    # Construir contexto con los mejores chunks
                    best_chunk_ids = [chunk.get('chunk_id') for chunk in best_chunks]
                    context_result = await self.context_builder.build_context(
                        chunk_ids=best_chunk_ids,
                        strategy=strategy,
                        max_chunks=top_k
                    )
                    
                    retrieval_results['chunks'] = context_result.get('chunks', [])
                    retrieval_results['context'] = context_result.get('context', '')
                else:
                    retrieval_results['chunks'] = []
                    retrieval_results['context'] = f"No suitable information found for: '{query_text}'"
        else:
            # Si no hay resultados en búsqueda vectorial, añadir lista vacía
            retrieval_results['chunks'] = []
        
        # Si no hay resultados, intentar búsqueda por palabras clave como fallback
        if not retrieval_results.get('chunks') and hasattr(self.embedding_repository, 'search_by_keywords'):
            logger.info(f"No vector search results, trying keyword search for: {query_text}")
            keyword_results = await self.embedding_repository.search_by_keywords(
                keywords=query_text,
                limit=top_k * 2  # Solicitar más para filtrar
            )
            
            if keyword_results:
                # Convertir resultados de palabras clave a formato de chunks
                keyword_chunks = [
                    {
                        'chunk_id': result.get('chunk_id'),
                        'content': result.get('content', ''),
                        'title': result.get('title', 'Unknown'),
                        'doc_id': result.get('doc_id'),
                        'similarity': result.get('relevance', 0.5)  # Usar relevancia como similitud
                    }
                    for result in keyword_results
                ]
                
                # Calcular info_score para cada chunk
                for chunk in keyword_chunks:
                    chunk['info_score'] = calculate_info_score(chunk.get('content', ''))
                
                # Filtrar, eliminar duplicados y diversificar
                filtered_keyword_chunks = filter_low_quality_chunks(keyword_chunks, mode='keyword')
                unique_keyword_chunks = remove_duplicate_chunks(filtered_keyword_chunks)
                ranked_chunks = combine_and_rank_chunks(unique_keyword_chunks, max_results=top_k)
                
                if ranked_chunks:
                    # Si hay chunks después de filtrado, construir contexto
                    keyword_chunk_ids = [chunk.get('chunk_id') for chunk in ranked_chunks]
                    keyword_context = await self.context_builder.build_context(
                        chunk_ids=keyword_chunk_ids,
                        strategy=strategy,
                        max_chunks=top_k
                    )
                    
                    retrieval_results['chunks'] = keyword_context.get('chunks', [])
                    retrieval_results['context'] = keyword_context.get('context', '')
                    logger.info(f"Found {len(retrieval_results['chunks'])} results via keyword search")
                else:
                    logger.info("No valid chunks after keyword search filtering")
        
        # Si todavía no hay contexto, crear un mensaje de no resultados
        if 'context' not in retrieval_results or not retrieval_results['context']:
            retrieval_results['context'] = f"No suitable information found for: '{query_text}'"
        
        # Asegurar que retrieval_results tiene la clave 'chunks'
        if 'chunks' not in retrieval_results:
            retrieval_results['chunks'] = []
            
        # Añadir metadatos de consulta
        retrieval_results['query'] = {
            'text': query_text,
            'processed': query_data['processed_query'],
            'language': query_data['language'],
            'type': query_data['query_type']
        }
        
        return retrieval_results
    
    def _combine_bilingual_results(self, original_results, translated_results, 
                                  source_lang, target_lang, top_k):
        """
        Combina resultados de búsqueda en diferentes idiomas en un único conjunto
        equilibrado, con etiquetas de idioma.
        
        Args:
            original_results: Resultados de la búsqueda con consulta original
            translated_results: Resultados de la búsqueda con consulta traducida
            source_lang: Idioma original de la consulta (en, es)
            target_lang: Idioma alternativo (en, es)
            top_k: Número máximo de resultados a devolver
            
        Returns:
            Diccionario con resultados combinados
        """
        # Obtener chunks de ambos resultados
        original_chunks = original_results.get('chunks', [])
        translated_chunks = translated_results.get('chunks', [])
        
        # Si alguno está vacío, devolver el otro
        if not original_chunks and not translated_chunks:
            return {
                'chunks': [],
                'context': f"No information found in either language for this query.",
                'query': original_results.get('query', {}),
                'search_metadata': {
                    'cross_lingual': True,
                    'languages': [source_lang, target_lang],
                    'result_count': 0
                }
            }
        elif not original_chunks:
            # Etiquetar chunks traducidos con su idioma
            for chunk in translated_chunks:
                chunk['source_language'] = target_lang
            return translated_results
        elif not translated_chunks:
            # Etiquetar chunks originales con su idioma
            for chunk in original_chunks:
                chunk['source_language'] = source_lang
            return original_results
        
        # Etiquetar cada chunk con su idioma de origen
        for chunk in original_chunks:
            chunk['source_language'] = source_lang
            
        for chunk in translated_chunks:
            chunk['source_language'] = target_lang
        
        # Combinar todos los chunks
        all_chunks = original_chunks + translated_chunks
        
        # Eliminar posibles duplicados cross-lingüísticos
        unique_chunks = deduplicate_bilingual_chunks(all_chunks)
        
        # Diversificar entre idiomas - máximo dos por idioma por documento
        lang_diverse_chunks = []
        
        # Agrupar por documento e idioma
        docs_by_lang = {
            source_lang: {},
            target_lang: {}
        }
        
        for chunk in unique_chunks:
            lang = chunk.get('source_language', source_lang)
            doc_id = chunk.get('doc_id', 'unknown')
            
            if doc_id not in docs_by_lang[lang]:
                docs_by_lang[lang][doc_id] = []
                
            docs_by_lang[lang][doc_id].append(chunk)
        
        # Priorizar un chunk de cada documento en el idioma original
        for doc_id, chunks in docs_by_lang[source_lang].items():
            # Ordenar por info_score o similarity
            if 'info_score' in chunks[0]:
                chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
            elif 'similarity' in chunks[0]:
                chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
            # Añadir el mejor chunk
            if chunks:
                lang_diverse_chunks.append(chunks[0])
        
        # Luego un chunk de cada documento en el idioma traducido
        for doc_id, chunks in docs_by_lang[target_lang].items():
            # Ordenar por info_score o similarity
            if 'info_score' in chunks[0]:
                chunks.sort(key=lambda x: x.get('info_score', 0), reverse=True)
            elif 'similarity' in chunks[0]:
                chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
            # Añadir el mejor chunk si aún no tenemos suficientes
            if chunks and len(lang_diverse_chunks) < top_k:
                lang_diverse_chunks.append(chunks[0])
        
        # Si aún necesitamos más chunks, añadir segundos mejores
        if len(lang_diverse_chunks) < top_k:
            # Segundos mejores del idioma original
            for doc_id, chunks in docs_by_lang[source_lang].items():
                if len(chunks) > 1:
                    lang_diverse_chunks.append(chunks[1])
                    if len(lang_diverse_chunks) >= top_k:
                        break
            
            # Segundos mejores del idioma traducido
            if len(lang_diverse_chunks) < top_k:
                for doc_id, chunks in docs_by_lang[target_lang].items():
                    if len(chunks) > 1:
                        lang_diverse_chunks.append(chunks[1])
                        if len(lang_diverse_chunks) >= top_k:
                            break
        
        # Limitar al top_k final
        final_chunks = lang_diverse_chunks[:top_k]
        
        # Construir contexto combinado con etiquetas de idioma
        context_parts = []
        for chunk in final_chunks:
            lang_indicator = "[ES]" if chunk.get('source_language') == 'es' else "[EN]"
            doc_title = chunk.get('title', 'Unknown')
            content = chunk.get('content', '')
            
            context_parts.append(f"{lang_indicator} [Documento: {doc_title}]\n{content}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Construir resultado final
        result = {
            'chunks': final_chunks,
            'context': combined_context,
            'query': {
                'text': original_results['query']['text'],
                'language': source_lang,
                'type': original_results['query']['type'],
                'translated_text': translated_results['query']['text'],
                'translated_language': target_lang
            },
            'search_metadata': {
                'mode': original_results.get('search_metadata', {}).get('mode', 'hybrid'),
                'strategy': original_results.get('search_metadata', {}).get('strategy', 'relevance'),
                'top_k': top_k,
                'result_count': len(final_chunks),
                'cross_lingual': True,
                'languages': [source_lang, target_lang]
            }
        }
        
        return result
    
    async def search(self, query_text: str, mode: str = 'hybrid', top_k: int = 5, strategy: str = 'relevance') -> Dict[str, Any]:
        """
        Perform a search in the RAG system.
        
        Args:
            query_text: Query text
            mode: Search mode ('faiss', 'pgvector', 'hybrid')
            top_k: Number of results to return
            strategy: Strategy for context building ('relevance', 'chronological', 'by_document')
            
        Returns:
            Search results including context and chunks
        """
        logger.info(f"Performing search: '{query_text}' (mode: {mode}, top_k: {top_k})")
        logger.debug(f"Iniciando búsqueda con query: '{query_text}', modo: {mode}")
        start_time = time.time()
        
        # Process query
        query_data = await self.query_processor.process_query(query_text)
        source_language = query_data['language']
        logger.info(f"Query processed - Language: {source_language}, Type: {query_data['query_type']}")
        
        # Verificar si podemos hacer búsqueda cruzada
        cross_lingual_search = (
            hasattr(self.query_processor, 'enable_cross_lingual') and 
            self.query_processor.enable_cross_lingual and
            'translated_query' in query_data
        )
        
        # Perform search with vector
        try:
            # Si está habilitada la búsqueda cruzada entre idiomas
            if cross_lingual_search:
                # Determinar idioma alterno
                target_language = query_data['alternate_language']
                translated_query = query_data['translated_query']
                
                logger.info(f"Cross-lingual search enabled. Original: '{query_text}' ({source_language}), "
                           f"Translated: '{translated_query}' ({target_language})")
                
                # Crear datos de consulta para la traducción
                translated_data = {
                    'processed_query': translated_query,
                    'language': target_language,
                    'query_type': query_data['query_type'],
                    'embedding': query_data['embedding'],  # Podríamos generar uno específico para el idioma traducido
                    'embedding_model': query_data['embedding_model']
                }
                
                # Ejecutar búsquedas en paralelo
                try:
                    # Búsqueda con consulta original
                    original_task = asyncio.create_task(
                        self._execute_search(query_data, mode, top_k, query_text, strategy)
                    )
                    
                    # Búsqueda con consulta traducida
                    translated_task = asyncio.create_task(
                        self._execute_search(translated_data, mode, top_k, translated_query, strategy)
                    )
                    
                    # Esperar resultados de ambas búsquedas
                    original_result, translated_result = await asyncio.gather(original_task, translated_task)
                    
                    # Combinar resultados
                    combined_result = self._combine_bilingual_results(
                        original_result, translated_result, 
                        source_language, target_language, 
                        top_k
                    )
                    
                    duration = time.time() - start_time
                    combined_result['search_metadata']['execution_time'] = duration
                    
                    logger.info(f"Cross-lingual search completed in {duration:.2f}s: {len(combined_result['chunks'])} chunks found "
                              f"({source_language}: {len(original_result.get('chunks', []))}, "
                              f"{target_language}: {len(translated_result.get('chunks', []))})")
                    
                    return combined_result
                    
                except Exception as e:
                    logger.error(f"Error in cross-lingual search: {e}")
                    logger.error(traceback.format_exc())
                    # Fallback to standard search if cross-lingual fails
            
            # Búsqueda estándar (sin traducción)
            result = await self._execute_search(query_data, mode, top_k, query_text, strategy)
            
            # Añadir metadatos de búsqueda
            result['search_metadata'] = {
                'mode': mode,
                'strategy': strategy, 
                'top_k': top_k,
                'execution_time': time.time() - start_time,
                'result_count': len(result.get('chunks', []))
            }
            
            duration = time.time() - start_time
            logger.info(f"Search completed in {duration:.2f}s: {len(result.get('chunks', []))} chunks found")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {mode} search: {e}")
            logger.error(traceback.format_exc())
            return {'chunks': [], 'context': f"Error searching for information: '{query_text}'"}

async def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search Pipeline')
    parser.add_argument('--query', type=str, required=True,
                      help='Search query')
    parser.add_argument('--mode', choices=['faiss', 'pgvector', 'hybrid'],
                      default='hybrid', help='Search mode')
    parser.add_argument('--top-k', type=int, default=5,
                      help='Number of results to return')
    parser.add_argument('--strategy', choices=['relevance', 'chronological', 'by_document'],
                      default='relevance', help='Strategy for context building')
    parser.add_argument('--use-container-name', action='store_true',
                      help='Use postgres_pgvector instead of localhost')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--no-cross-lingual', action='store_true',
                      help='Disable cross-lingual search')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    config = Config()
    if args.no_cross_lingual:
        config.ENABLE_CROSS_LINGUAL = False
        
    pipeline = SearchPipeline(config=config, use_localhost=not args.use_container_name)
    try:
        result = await pipeline.search(args.query, mode=args.mode, top_k=args.top_k, strategy=args.strategy)
        
        print(f"\n{'='*50}")
        print("SEARCH RESULTS")
        print(f"{'='*50}")
        print(f"Query: '{args.query}'")
        
        # Mostrar consulta traducida si está disponible
        if result.get('query', {}).get('translated_text'):
            print(f"Translated: '{result['query']['translated_text']}'")
            print(f"Languages: {result.get('search_metadata', {}).get('languages', ['unknown'])}")
            
        print(f"Mode: {args.mode}")
        print(f"Strategy: {args.strategy}")
        print(f"Found {len(result.get('chunks', []))} relevant chunks")
        
        if result.get('context'):
            print("\nContext:")
            print(f"{'='*50}")
            print(result['context'])
            print(f"{'='*50}")
        else:
            print("\nNo context could be generated")
            
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())