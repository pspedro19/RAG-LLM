# app/query/context_builder.py

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import re

from app.core.config.config import Config
from app.ingestion.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Construye un contexto enriquecido a partir de los chunks recuperados.
    Proporciona diferentes estrategias para formatear y ordenar los chunks.
    """
    
    def __init__(self, config: Config, chunk_repository: ChunkRepository):
        """
        Inicializa el constructor de contexto.
        
        Args:
            config: Configuración del sistema
            chunk_repository: Repositorio para acceder a los chunks
        """
        self.config = config
        self.chunk_repository = chunk_repository
        
        # Configuración
        self.max_context_length = 4000
        self.separator = '\n\n'
        self.add_metadata = True
        self.default_strategy = 'relevance'
        
        # Patrones para detección de contenido estructural vs. informativo
        self.toc_patterns = [
            re.compile(r'(?i)(table of contents|índice|contents|contenidos)'),
            re.compile(r'(?i)(list of (tables|figures)|lista de (tablas|figuras))'),
            re.compile(r'(?i)(bibliography|references|bibliografía|referencias)'),
            re.compile(r'(?i)(appendix|apéndice)')
        ]
        
        # Patrones para detección de alta densidad informativa
        self.informative_patterns = [
            re.compile(r'\d+%'),  # Porcentajes
            re.compile(r'(?i)key (finding|result)s?:'),  # Hallazgos clave
            re.compile(r'(?i)conclusion'),  # Conclusiones
            re.compile(r'(?i)importantly'),  # Marcadores de importancia
        ]
    
    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Recupera información completa de chunks a partir de sus IDs.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            
        Returns:
            Lista de datos completos de chunks
        """
        if not chunk_ids:
            return []
        
        chunks = []
        for chunk_id in chunk_ids:
            if not chunk_id:  # Ignorar IDs nulos
                continue
                
            chunk = await self.chunk_repository.get_chunk_by_id(chunk_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def deduplicate_chunks(self, chunks: List[Dict[str, Any]], similarities: List[float] = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Remove duplicate chunks based on content similarity.
        Usa un umbral más alto (0.98) para ser menos agresivo en la deduplicación.
        
        Args:
            chunks: List of chunks
            similarities: Optional list of similarity scores
            
        Returns:
            Tuple of (deduplicated chunks, corresponding similarities)
        """
        if not chunks:
            return [], []
        
        # Usar un diccionario para rastrear contenido visto por hash
        seen_content = {}
        unique_chunks = []
        unique_similarities = []
        
        for i, chunk in enumerate(chunks):
            # Obtener contenido y crear un hash
            content = chunk.get('content', '')
            content_hash = hash(content)
            
            # Verificar primero duplicados exactos (mismo hash)
            if content_hash in seen_content:
                logger.debug(f"Found exact duplicate chunk (same hash): {content[:50]}...")
                continue
                
            # Verificar contenido casi duplicado con un umbral más alto (98% de solapamiento)
            is_duplicate = False
            for seen_hash, seen_content_text in seen_content.items():
                # Solo comparar longitud de texto para contenidos de longitud muy similar
                if abs(len(content) - len(seen_content_text)) / max(len(content), len(seen_content_text)) > 0.2:
                    # Si la diferencia de longitud es mayor al 20%, no es probable que sea duplicado
                    continue
                    
                # Calcular similitud (versión básica - puede mejorarse)
                shorter, longer = (content, seen_content_text) if len(content) < len(seen_content_text) else (seen_content_text, content)
                
                # Verificar primero si hay coincidencia de frases clave (más rápido que comparar todo el texto)
                # Extraer primeras 2-3 frases de cada texto para comparación rápida
                first_sentences_pattern = re.compile(r'([^.!?]+[.!?]){1,3}')
                shorter_intro = first_sentences_pattern.match(shorter)
                longer_intro = first_sentences_pattern.match(longer)
                
                # Si las primeras frases son muy diferentes, probablemente no son duplicados
                if shorter_intro and longer_intro:
                    shorter_intro = shorter_intro.group(0)
                    longer_intro = longer_intro.group(0)
                    intro_similarity = len(shorter_intro) / len(longer_intro) if len(longer_intro) > 0 else 0
                    if intro_similarity < 0.7:
                        continue
                
                # Calcular similitud completa
                similarity = len(shorter) / len(longer) if len(longer) > 0 else 0
                
                # Usar un umbral mucho más alto (0.98) para considerar como duplicado
                if similarity > 0.98:  # Cambiado de 0.95 a 0.98 para ser menos agresivo
                    is_duplicate = True
                    logger.debug(f"Found near-duplicate chunk: {content[:50]}...")
                    break
            
            if not is_duplicate:
                seen_content[content_hash] = content
                unique_chunks.append(chunk)
                if similarities and i < len(similarities):
                    unique_similarities.append(similarities[i])
        
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        
        if similarities:
            return unique_chunks, unique_similarities
        return unique_chunks, []
    
    def calculate_informative_score(self, content: str) -> float:
        """
        Calcula un puntaje de informatividad para un contenido.
        Los puntajes más altos indican contenido más informativo.
        
        Args:
            content: Texto a analizar
            
        Returns:
            Puntaje de informatividad (0.0 a 1.0)
        """
        if not content or len(content) < 50:
            return 0.1
            
        # Factores positivos
        positive_factors = []
        
        # 1. Longitud del contenido (más largo suele ser más informativo)
        length_score = min(1.0, len(content) / 1000)
        positive_factors.append(length_score)
        
        # 2. Presencia de datos numéricos y porcentajes
        percentage_count = len(re.findall(r'\d+%|\d+\.\d+%', content))
        number_count = sum(c.isdigit() for c in content)
        number_density = number_count / len(content) if content else 0
        
        # Ajustar para penalizar TOCs (que tienen muchos números pero no son informativos)
        if percentage_count > 0:
            number_score = min(1.0, 0.5 + percentage_count / 10)  # Bonificación por porcentajes
        elif 0.05 <= number_density <= 0.15:  # Rango saludable para texto informativo
            number_score = 0.7
        else:
            number_score = max(0.1, min(0.6, number_density * 4))  # Penalización para valores extremos
            
        positive_factors.append(number_score)
        
        # 3. Presencia de patrones informativos
        info_pattern_matches = sum(1 for p in self.informative_patterns if p.search(content))
        info_pattern_score = min(1.0, info_pattern_matches / 2)
        positive_factors.append(info_pattern_score)
        
        # 4. Densidad de texto (proporción de palabras vs. caracteres)
        words = re.findall(r'\b\w+\b', content)
        word_density = len(words) / len(content) if content else 0
        word_density_score = min(1.0, word_density * 7)  # Normalizar a escala 0-1
        positive_factors.append(word_density_score)
        
        # Factores negativos
        negative_factors = []
        
        # 1. Muchos puntos suspensivos (típico en TOCs)
        ellipsis_density = content.count('...') / len(content) if content else 0
        ellipsis_penalty = max(0, min(0.8, ellipsis_density * 50))
        negative_factors.append(ellipsis_penalty)
        
        # 2. Líneas cortas repetitivas (típico en índices)
        lines = content.split('\n')
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
    
    def is_toc_section(self, text: str) -> bool:
        """
        Detecta si un texto parece ser una tabla de contenido, índice o lista de figuras/tablas.
        Mejorado para mejor discriminación entre TOCs y contenido valioso.
        
        Args:
            text: Texto a analizar
            
        Returns:
            True si parece una tabla de contenido, False en caso contrario
        """
        # Logging del texto para diagnóstico
        if len(text) > 100:
            logger.debug(f"ContextBuilder TOC check - texto: {text[:100]}... (len: {len(text)})")
        else:
            logger.debug(f"ContextBuilder TOC check - texto: {text}")
        
        # Verificar títulos explícitos de TOC
        toc_indicators = [
            "Table of Contents", "Contents", "List of Tables", "List of Figures",
            "Índice", "Contenidos", "Lista de Tablas", "Lista de Figuras"
        ]
        
        # Verificar marcadores de título (coincidencias exactas o al principio de línea)
        for indicator in toc_indicators:
            # Buscar coincidencia exacta como título
            if re.search(r'(^|\n)[ \t]*' + re.escape(indicator) + r'[ \t]*($|\n)', text, re.IGNORECASE):
                logger.debug(f"ContextBuilder: TOC por título exacto: '{indicator}'")
                return True
        
        # Verificar si hay muchas entradas "Figure X" o "Table X" seguidas
        figure_table_pattern = re.compile(r'(Figure|Table|Figura|Tabla)\s+\d+')
        matches = figure_table_pattern.findall(text)
        if len(matches) > 3 and len(matches) / (text.count('\n') + 1) > 0.4:
            logger.debug(f"ContextBuilder: Detectado como lista de figuras/tablas ({len(matches)} ocurrencias)")
            return True
        
        # Excepción específica para tablas de datos que no son TOC
        if re.search(r'Table \d+:', text):
            dots_ratio = text.count('.') / len(text) if text else 0
            if dots_ratio < 0.1:  # Pocos puntos (típico en tablas de datos, no en TOC)
                logger.debug("ContextBuilder: Contiene 'Table X:' pero parece ser una tabla de datos, no un TOC")
                return False
        
        # Verificar formato tipo TOC
        lines = text.split('\n')
        toc_line_count = 0
        
        for i, line in enumerate(lines[:15]):  # Verificar más líneas (15 en lugar de 10)
            # Hacer la detección más restrictiva
            if (re.search(r'\d+\.\d+', line) and re.search(r'\.{3,}\s*\d+$', line)) or \
               (re.search(r'^\s*[A-Z][^.]+\s+\.{3,}\s*\d+$', line)):  # Formato "Capítulo ... 10"
                toc_line_count += 1
                logger.debug(f"ContextBuilder: Línea {i+1} con formato TOC: '{line}'")
        
        toc_line_ratio = toc_line_count / len(lines) if len(lines) > 0 else 0
        logger.debug(f"ContextBuilder: Ratio líneas TOC: {toc_line_count}/{len(lines)} = {toc_line_ratio:.3f}")
        
        # Umbral ajustado (40% en lugar de 50%)
        if len(lines) > 5 and toc_line_ratio > 0.4:
            logger.debug(f"ContextBuilder: TOC por formato de líneas: ratio {toc_line_ratio:.3f} > 0.4")
            return True
            
        # Verificar densidad de puntos y números
        dots_count = text.count('.')
        numbers_count = sum(c.isdigit() for c in text)
        ratio = (dots_count + numbers_count) / len(text) if text else 0
        
        logger.debug(f"ContextBuilder: Densidad puntos/números: {ratio:.4f} (puntos: {dots_count}, números: {numbers_count}, total: {len(text)})")
        
        # Verificar si hay muchos puntos suspensivos (típico en TOC)
        ellipsis_count = text.count('...')
        if ellipsis_count > 3 and ellipsis_count / len(lines) > 0.2:
            logger.debug(f"ContextBuilder: Alto número de puntos suspensivos: {ellipsis_count}")
            return True
        
        # Umbral ajustado (22% en lugar de 25%)
        if ratio > 0.22 and len(lines) > 5:
            logger.debug(f"ContextBuilder: TOC por densidad puntos/números: {ratio:.4f} > 0.22")
            return True
        
        logger.debug("ContextBuilder: No detectado como TOC")
        return False
        
    def format_chunk_metadata(self, chunk: Dict[str, Any]) -> str:
        """
        Formatea los metadatos de un chunk para incluirlos en el contexto.
        
        Args:
            chunk: Datos del chunk
            
        Returns:
            Texto formateado con metadatos
        """
        if not self.add_metadata:
            return ""
        
        metadata_lines = []
        
        # Añadir título del documento
        if 'title' in chunk and chunk['title']:
            metadata_lines.append(f"Documento: {chunk['title']}")
        
        # Añadir metadatos específicos si existen
        if 'metadata' in chunk and chunk['metadata']:
            meta = chunk['metadata']
            
            # Añadir categoría si existe
            if 'category' in meta:
                metadata_lines.append(f"Categoría: {meta['category']}")
                
            # Añadir otra información relevante
            for key in ['author', 'date', 'source', 'section']:
                if key in meta and meta[key]:
                    metadata_lines.append(f"{key.capitalize()}: {meta[key]}")
        
        # Formatear como bloque
        if metadata_lines:
            return f"[{' | '.join(metadata_lines)}]\n"
        return ""
    
    def format_chunk(self, chunk: Dict[str, Any], include_metadata: bool = None) -> str:
        """
        Formatea un chunk para incluirlo en el contexto.
        
        Args:
            chunk: Datos del chunk
            include_metadata: Si se deben incluir metadatos
            
        Returns:
            Texto formateado del chunk
        """
        include_metadata = self.add_metadata if include_metadata is None else include_metadata
        
        # Obtener contenido
        content = chunk.get('content', '')
        
        # Añadir metadatos si corresponde
        if include_metadata:
            metadata = self.format_chunk_metadata(chunk)
            return f"{metadata}{content}"
        
        return content
    
    def truncate_context(self, context: str, max_length: int = None) -> str:
        """
        Trunca el contexto si excede la longitud máxima.
        
        Args:
            context: Contexto completo
            max_length: Longitud máxima (None para usar el valor predeterminado)
            
        Returns:
            Contexto truncado
        """
        max_length = max_length or self.max_context_length
        
        if len(context) <= max_length:
            return context
        
        # Encontrar un buen punto para truncar (fin de oración)
        truncate_point = max_length
        
        # Buscar el último punto dentro del límite permitido
        last_period = context.rfind('.', 0, max_length)
        if last_period > max_length * 0.8:  # Si el punto está en el último 20%
            truncate_point = last_period + 1
        
        return context[:truncate_point] + f"... [Contexto truncado de {len(context)} a {truncate_point} caracteres]"
    
    def filter_low_quality_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra chunks de baja calidad como TOCs, chunks muy cortos, etc.
        Mejorado para mejor detección de contenido informativo vs. estructural.
        
        Args:
            chunks: Lista de chunks a filtrar
            
        Returns:
            Lista filtrada de chunks de alta calidad
        """
        if not chunks:
            return []
            
        filtered_chunks = []
        filter_reasons = {"short": 0, "toc": 0, "low_info": 0, "other": 0}
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Filtrar contenido muy corto
            if len(content) < 50:  # Reducido de 100 a 50
                filter_reasons["short"] += 1
                logger.debug(f"Skipping very short chunk ({len(content)} chars): {content[:30]}...")
                continue
            
            # Filtrar secciones de TOC y referencias
            if self.is_toc_section(content):
                filter_reasons["toc"] += 1
                logger.debug(f"Skipping TOC section: {content[:50]}...")
                continue
            
            # Calcular score informativo y filtrar si es muy bajo
            info_score = self.calculate_informative_score(content)
            if info_score < 0.3:  # Umbral ajustable para mantener solo contenido informativo
                filter_reasons["low_info"] += 1
                logger.debug(f"Skipping low information content (score: {info_score:.2f}): {content[:50]}...")
                continue
            
            filtered_chunks.append(chunk)
        
        # Registrar razones específicas de filtrado
        if sum(filter_reasons.values()) > 0:
            logger.info(f"Filtered chunks by reason: short={filter_reasons['short']}, toc={filter_reasons['toc']}, low_info={filter_reasons['low_info']}")
        
        logger.info(f"Filtered {len(chunks) - len(filtered_chunks)} low quality chunks from {len(chunks)} total")
        
        # Asegurar un mínimo de chunks (incluso si son de baja calidad)
        if len(filtered_chunks) == 0 and len(chunks) > 0:
            # Si todos fueron filtrados, devolver al menos el que tenga más texto
            chunks_by_length = sorted(chunks, key=lambda x: len(x.get('content', '')), reverse=True)
            filtered_chunks = [chunks_by_length[0]]
            logger.info("All chunks were filtered. Returning at least one chunk despite quality concerns")
        
        return filtered_chunks
    
    def diversify_chunks(self, chunks: List[Dict[str, Any]], max_per_doc: int = 2) -> List[Dict[str, Any]]:
        """
        Diversifica chunks para no sobrerrepresentar un solo documento.
        
        Args:
            chunks: Lista de chunks a diversificar
            max_per_doc: Máximo número de chunks por documento
            
        Returns:
            Lista diversificada de chunks
        """
        if not chunks or max_per_doc <= 0:
            return chunks
            
        # Agrupar chunks por documento
        chunks_by_doc = {}
        for chunk in chunks:
            doc_id = chunk.get('doc_id')
            if not doc_id:
                continue
                
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)
        
        # Si no hay documentos múltiples, devolver chunks originales
        if len(chunks_by_doc) <= 1:
            return chunks
            
        # Seleccionar hasta max_per_doc chunks de cada documento
        diverse_chunks = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            # Ordenar por score informativo (si está disponible en metadata)
            try:
                sorted_chunks = sorted(
                    doc_chunks,
                    key=lambda x: self.calculate_informative_score(x.get('content', '')),
                    reverse=True
                )
            except:
                # Si hay error en el cálculo, usar el orden original
                sorted_chunks = doc_chunks
                
            # Tomar los mejores chunks (hasta max_per_doc)
            diverse_chunks.extend(sorted_chunks[:max_per_doc])
        
        if len(diverse_chunks) < len(chunks):
            logger.info(f"Diversified chunks: {len(chunks)} -> {len(diverse_chunks)} (from {len(chunks_by_doc)} documents)")
            
        return diverse_chunks
    
    async def build_context_by_relevance(
        self, 
        chunk_ids: List[str], 
        similarities: List[float] = None, 
        max_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Construye un contexto ordenando los chunks por relevancia.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            similarities: Lista opcional de puntuaciones de similitud
            max_chunks: Número máximo de chunks a incluir
            
        Returns:
            Diccionario con contexto y metadatos
        """
        if not chunk_ids:
            return {"context": "", "chunks": [], "strategy": "relevance"}
        
        # Obtener datos completos de chunks
        chunks = await self.get_chunks_by_ids(chunk_ids)
        
        # Filtrar chunks de baja calidad
        chunks = self.filter_low_quality_chunks(chunks)
        
        # Remove duplicates (less aggressively)
        if similarities and len(similarities) == len(chunks):
            chunks, similarities = self.deduplicate_chunks(chunks, similarities)
        else:
            chunks, _ = self.deduplicate_chunks(chunks)
        
        # Si hay puntuaciones de similitud, ordenar por ellas
        if similarities and len(similarities) == len(chunks):
            chunks_with_scores = list(zip(chunks, similarities))
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_chunks = [chunk for chunk, _ in chunks_with_scores]
            sorted_scores = [score for _, score in chunks_with_scores]
        else:
            # Mantener el orden original (asumiendo que ya está por relevancia)
            sorted_chunks = chunks
            sorted_scores = similarities if similarities else [None] * len(chunks)
        
        # Diversificar para no sobrerrepresentar un solo documento
        if len(sorted_chunks) > 1:
            max_per_doc = max(1, max_chunks // 2) if max_chunks else 2
            sorted_chunks = self.diversify_chunks(sorted_chunks, max_per_doc)
            
            # Reajustar scores si se cambió el orden
            if sorted_scores and len(sorted_scores) == len(chunks):
                sorted_scores = [
                    sorted_scores[chunks.index(chunk)] if chunk in chunks else None
                    for chunk in sorted_chunks
                ]
        
        # Limitar número de chunks si es necesario
        if max_chunks and max_chunks < len(sorted_chunks):
            sorted_chunks = sorted_chunks[:max_chunks]
            sorted_scores = sorted_scores[:max_chunks]
        
        # Construir contexto
        context_parts = []
        for chunk in sorted_chunks:
            context_parts.append(self.format_chunk(chunk))
        
        # Unir con separador y truncar si es necesario
        full_context = self.separator.join(context_parts)
        truncated_context = self.truncate_context(full_context)
        
        # Preparar resultado con metadatos
        result = {
            "context": truncated_context,
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content", ""),  # Vista previa
                    "title": chunk.get("title", ""),
                    "doc_id": chunk.get("doc_id"),
                    "similarity": score,
                    "info_score": self.calculate_informative_score(chunk.get("content", ""))
                }
                for chunk, score in zip(sorted_chunks, sorted_scores)
            ],
            "strategy": "relevance",
            "is_truncated": len(full_context) > self.max_context_length,
            "original_length": len(full_context),
            "final_length": len(truncated_context)
        }
        
        return result
    
    async def build_context_chronological(
        self, 
        chunk_ids: List[str], 
        max_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Construye un contexto ordenando los chunks cronológicamente.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            max_chunks: Número máximo de chunks a incluir
            
        Returns:
            Diccionario con contexto y metadatos
        """
        if not chunk_ids:
            return {"context": "", "chunks": [], "strategy": "chronological"}
        
        # Obtener datos completos de chunks
        chunks = await self.get_chunks_by_ids(chunk_ids)
        
        # Filtrar chunks de baja calidad
        chunks = self.filter_low_quality_chunks(chunks)
        
        # Remove duplicates
        chunks, _ = self.deduplicate_chunks(chunks)
        
        # Diversificar para no sobrerrepresentar un solo documento
        if len(chunks) > 1:
            max_per_doc = max(1, max_chunks // 2) if max_chunks else 2
            diverse_chunks = self.diversify_chunks(chunks, max_per_doc)
            
            # Solo usar diversificación si no reduce demasiado el número de chunks
            if len(diverse_chunks) >= len(chunks) * 0.5:
                chunks = diverse_chunks
        
        # Agrupar por documento y ordenar por número de chunk
        chunks_by_doc = {}
        for chunk in chunks:
            doc_id = chunk.get('doc_id')
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)
        
        # Ordenar chunks dentro de cada documento
        for doc_id in chunks_by_doc:
            chunks_by_doc[doc_id].sort(key=lambda c: c.get('chunk_number', 0))
        
        # Aplanar la lista manteniendo el orden por documento
        sorted_chunks = []
        for doc_chunks in chunks_by_doc.values():
            sorted_chunks.extend(doc_chunks)
        
        # Limitar número de chunks si es necesario
        if max_chunks and max_chunks < len(sorted_chunks):
            sorted_chunks = sorted_chunks[:max_chunks]
        
        # Construir contexto
        context_parts = []
        for chunk in sorted_chunks:
            context_parts.append(self.format_chunk(chunk))
        
        # Unir con separador y truncar si es necesario
        full_context = self.separator.join(context_parts)
        truncated_context = self.truncate_context(full_context)
        
        # Preparar resultado con metadatos
        result = {
            "context": truncated_context,
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content", "")[:100] + "...",  # Vista previa
                    "title": chunk.get("title", ""),
                    "doc_id": chunk.get("doc_id"),
                    "chunk_number": chunk.get("chunk_number", 0),
                    "info_score": self.calculate_informative_score(chunk.get("content", ""))
                }
                for chunk in sorted_chunks
            ],
            "strategy": "chronological",
            "is_truncated": len(full_context) > self.max_context_length,
            "original_length": len(full_context),
            "final_length": len(truncated_context)
        }
        
        return result
    
    async def build_context_by_document(
        self, 
        chunk_ids: List[str], 
        similarities: List[float] = None, 
        max_chunks_per_doc: int = 2
    ) -> Dict[str, Any]:
        """
        Construye un contexto agrupando por documento y seleccionando los mejores chunks de cada uno.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            similarities: Lista opcional de puntuaciones de similitud
            max_chunks_per_doc: Número máximo de chunks por documento
            
        Returns:
            Diccionario con contexto y metadatos
        """
        if not chunk_ids:
            return {"context": "", "chunks": [], "strategy": "by_document"}
        
        # Obtener datos completos de chunks
        chunks = await self.get_chunks_by_ids(chunk_ids)
        
        # Filtrar chunks de baja calidad
        chunks = self.filter_low_quality_chunks(chunks)
        
        # Remove duplicates
        if similarities and len(similarities) == len(chunks):
            chunks, similarities = self.deduplicate_chunks(chunks, similarities)
        else:
            chunks, _ = self.deduplicate_chunks(chunks)
            
        # Preparar scores
        scores = similarities if similarities and len(similarities) == len(chunks) else [1.0] * len(chunks)
        
        # Agrupar chunks por documento con sus scores
        chunks_by_doc = {}
        for chunk, score in zip(chunks, scores):
            doc_id = chunk.get('doc_id')
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append((chunk, score))
        
        # Seleccionar los mejores chunks de cada documento
        selected_chunks = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            # Ordenar por relevancia
            doc_chunks.sort(key=lambda x: x[1], reverse=True)
            # Tomar los mejores N chunks
            best_chunks = doc_chunks[:max_chunks_per_doc]
            # Ordenarlos por número de chunk para mantener coherencia
            best_chunks.sort(key=lambda x: x[0].get('chunk_number', 0))
            # Añadir a la selección
            selected_chunks.extend(best_chunks)
        
        # Construir contexto
        context_parts = []
        for chunk, _ in selected_chunks:
            context_parts.append(self.format_chunk(chunk))
        
        # Unir con separador y truncar si es necesario
        full_context = self.separator.join(context_parts)
        truncated_context = self.truncate_context(full_context)
        
        # Preparar resultado con metadatos
        result = {
            "context": truncated_context,
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content", "")[:100] + "...",  # Vista previa
                    "title": chunk.get("title", ""),
                    "doc_id": chunk.get("doc_id"),
                    "similarity": score,
                    "info_score": self.calculate_informative_score(chunk.get("content", ""))
                }
                for chunk, score in selected_chunks
            ],
            "strategy": "by_document",
            "is_truncated": len(full_context) > self.max_context_length,
            "original_length": len(full_context),
            "final_length": len(truncated_context),
            "documents_count": len(chunks_by_doc)
        }
        
        return result
    
    async def build_context(
        self, 
        chunk_ids: List[str], 
        similarities: List[float] = None, 
        strategy: str = None,
        max_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Construye un contexto usando la estrategia especificada.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            similarities: Lista opcional de puntuaciones de similitud
            strategy: Estrategia a utilizar ('relevance', 'chronological', 'by_document')
            max_chunks: Número máximo de chunks a incluir
            
        Returns:
            Diccionario con contexto y metadatos
        """
        strategy = strategy or self.default_strategy
        
        if strategy == 'relevance':
            return await self.build_context_by_relevance(chunk_ids, similarities, max_chunks)
        elif strategy == 'chronological':
            return await self.build_context_chronological(chunk_ids, max_chunks)
        elif strategy == 'by_document':
            max_per_doc = max(1, max_chunks // 3) if max_chunks else 2
            return await self.build_context_by_document(chunk_ids, similarities, max_per_doc)
        else:
            logger.warning(f"Estrategia desconocida: {strategy}, usando relevance por defecto")
            return await self.build_context_by_relevance(chunk_ids, similarities, max_chunks)
    
    async def build_from_retrieval_results(
        self, 
        retrieval_results: Dict[str, Any], 
        strategy: str = None,
        max_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Construye un contexto a partir de resultados de un Retriever.
        
        Args:
            retrieval_results: Resultados de búsqueda del Retriever
            strategy: Estrategia de construcción de contexto
            max_chunks: Número máximo de chunks
            
        Returns:
            Diccionario con contexto construido y metadatos
        """
        # Extraer chunk_ids y similarities de los resultados
        chunk_ids = retrieval_results.get('chunk_ids', [])
        similarities = retrieval_results.get('similarities', [[]])[0] if retrieval_results.get('similarities') else None
        
        # Construir el contexto
        context_result = await self.build_context(
            chunk_ids=chunk_ids,
            similarities=similarities,
            strategy=strategy,
            max_chunks=max_chunks
        )
        
        # Añadir información de la consulta si existe
        if 'query' in retrieval_results:
            context_result['query'] = retrieval_results['query']
        
        # Añadir información de búsqueda
        context_result['search'] = {
            'time': retrieval_results.get('search_time'),
            'mode': retrieval_results.get('mode'),
            'k': retrieval_results.get('k')
        }
        
        # Añadir los resultados originales para compatibilidad
        if 'results' in retrieval_results:
            context_result['results'] = retrieval_results['results']
        
        return context_result