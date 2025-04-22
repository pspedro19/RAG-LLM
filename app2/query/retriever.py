# app2/query/retriever.py

import logging
import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Union
import time
import re

from app2.core.config.config import Config
from app2.core.faiss_manager import FAISSVectorStore
from app2.embeddings.embedding_repository import EmbeddingRepository

logger = logging.getLogger(__name__)

class Retriever:
    """
    Clase para recuperar documentos o chunks relevantes a una consulta.
    Soporta búsqueda mediante FAISS y/o pgvector con ajustes de relevancia inteligentes
    y diversificación de resultados.
    """
    
    def __init__(
        self, 
        faiss_store: FAISSVectorStore,
        embedding_repository: EmbeddingRepository,
        config: Config = None
    ):
        """
        Inicializa el recuperador.
        
        Args:
            faiss_store: Almacén vectorial FAISS
            embedding_repository: Repositorio de embeddings en PostgreSQL
            config: Configuración del sistema
        """
        self.faiss_store = faiss_store
        self.embedding_repository = embedding_repository
        self.config = config or Config()
        
        # Configuración de búsqueda
        self.default_k = 5
        self.min_similarity = 0.1  # Umbral más permisivo para capturar más resultados iniciales
        self.search_mode = 'hybrid'  # 'faiss', 'pgvector', o 'hybrid'
        
        # Configuración para filtrado de contenido
        self.toc_patterns = [
            re.compile(r'(?i)(table of contents|índice|contents|contenidos)'),
            re.compile(r'(?i)(list of (tables|figures)|lista de (tablas|figuras))'),
            re.compile(r'(?i)(bibliography|references|bibliografía|referencias)'),
            re.compile(r'(?i)(appendix|apéndice)')
        ]
        
        # Patrones de alta densidad informativa
        self.informative_patterns = [
            re.compile(r'\d+%'),  # Porcentajes
            re.compile(r'(?i)key (finding|result)s?:'),  # Hallazgos clave
            re.compile(r'(?i)conclusion'),  # Conclusiones
            re.compile(r'(?i)importantly'),  # Marcadores de importancia
        ]
    
    def is_toc_section(self, content: str) -> bool:
        """
        Detecta si un texto parece ser una tabla de contenido o un índice.
        Criterios mejorados para evitar falsos positivos con tablas de datos.
        
        Args:
            content: Texto a analizar
            
        Returns:
            True si parece un índice o TOC, False en caso contrario
        """
        if not content:
            return False
            
        # Logging del texto para diagnóstico
        if len(content) > 100:
            logger.debug(f"Retriever TOC check - texto: {content[:100]}... (len: {len(content)})")
        else:
            logger.debug(f"Retriever TOC check - texto: {content}")
            
        # Buscar patrones específicos de TOC
        for pattern in self.toc_patterns:
            match = pattern.search(content)
            if match:
                # Verificar que la coincidencia es un título exacto, no parte de otra frase
                matched_text = match.group(0)
                # Verificar si el match está al inicio o en una línea nueva
                if re.search(r'(^|\n)[ \t]*' + re.escape(matched_text) + r'[ \t]*($|\n)', content, re.IGNORECASE):
                    logger.debug(f"Retriever: TOC detectado por patrón regex exacto")
                    return True
                
        # Excepción específica para tablas de datos que no son TOC
        if re.search(r'Table \d+:', content):
            dots_ratio = content.count('.') / len(content) if content else 0
            if dots_ratio < 0.1:  # Pocos puntos (típico en tablas de datos, no en TOC)
                logger.debug("Retriever: Contiene 'Table X:' pero parece ser una tabla de datos, no un TOC")
                return False
                
        # Verificar si hay muchas entradas "Figure X" o "Table X" seguidas
        figure_table_pattern = re.compile(r'(Figure|Table|Figura|Tabla)\s+\d+')
        matches = figure_table_pattern.findall(content)
        if len(matches) > 3 and len(matches) / (content.count('\n') + 1) > 0.4:
            logger.debug(f"Retriever: Detectado como lista de figuras/tablas ({len(matches)} ocurrencias)")
            return True
                
        # Verificar formato con puntos y números (típico de TOC)
        lines = content.split('\n')
        toc_line_count = 0
        
        for i, line in enumerate(lines[:15]):  # Examinar más líneas (15 en lugar de 10)
            # Detectar líneas con formato TOC (más restrictivo)
            if (re.search(r'\d+\.\d+', line) and re.search(r'\.{3,}\s*\d+$', line)) or \
               (re.search(r'^\s*[A-Z][^.]+\s+\.{3,}\s*\d+$', line)):  # Formato "Capítulo ... 10"
                toc_line_count += 1
                logger.debug(f"Retriever: Línea {i+1} con formato TOC: '{line}'")
                
        toc_line_ratio = toc_line_count / len(lines) if len(lines) > 0 else 0
        logger.debug(f"Retriever: Ratio líneas TOC: {toc_line_ratio:.3f}")
                
        # Aumentar el umbral del 30% al 40%
        if len(lines) > 5 and toc_line_ratio > 0.4:
            logger.debug(f"Retriever: TOC por ratio líneas: {toc_line_ratio:.3f} > 0.4")
            return True
            
        # Contar caracteres específicos de TOC
        dots_count = content.count('.')
        numbers_count = sum(c.isdigit() for c in content)
        
        # Calcular proporción de puntos y números
        if len(content) > 0:
            ratio = (dots_count + numbers_count) / len(content)
            logger.debug(f"Retriever: Densidad puntos/números: {ratio:.4f} (puntos: {dots_count}, números: {numbers_count})")
            
            # Verificar si hay muchos puntos suspensivos (típico en TOC)
            ellipsis_count = content.count('...')
            if ellipsis_count > 3 and ellipsis_count / len(lines) > 0.2:
                logger.debug(f"Retriever: Alto número de puntos suspensivos: {ellipsis_count}")
                return True
                
            # Umbral ajustado (del 25% al 22%)
            if ratio > 0.22 and len(lines) > 5:
                logger.debug(f"Retriever: TOC por densidad puntos/números: {ratio:.4f} > 0.22")
                return True
        
        logger.debug("Retriever: No detectado como TOC")
        return False
    
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
    
    async def adjust_relevance_scores(self, chunk_ids: List[str], similarities: List[float], query_text: str = None) -> List[float]:
        """
        Ajusta puntuaciones de relevancia basadas en calidad del contenido y contexto.
        Penaliza TOCs y contenido poco informativo, premia contenido sustantivo.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            similarities: Lista de puntuaciones de similitud
            query_text: Texto original de la consulta para boosting contextual
            
        Returns:
            Puntuaciones de similitud ajustadas
        """
        if not chunk_ids or not similarities:
            return similarities
        
        # Obtener chunks completos para analizar contenido
        pool = await self.embedding_repository._get_pool()
        adjusted_scores = similarities.copy()
        
        # Extraer términos clave de la consulta
        query_terms = set()
        if query_text:
            # Extracción mejorada de términos significativos
            query_text = query_text.lower()
            # Remover stopwords y dividir en términos
            stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'about', 
                         'what', 'where', 'when', 'how', 'who', 'which', 'why', 'is', 'are', 
                         'do', 'does', 'can', 'could', 'would', 'should', 'will'}
            query_terms = {term for term in re.findall(r'\b\w+\b', query_text) 
                          if term not in stop_words and len(term) > 2}
        
        async with pool.acquire() as conn:
            for i, chunk_id in enumerate(chunk_ids):
                if not chunk_id or i >= len(adjusted_scores):
                    continue
                    
                # Obtener contenido del chunk
                row = await conn.fetchrow("""
                    SELECT content, metadata
                    FROM chunks
                    WHERE chunk_id = $1
                """, chunk_id)
                
                if not row:
                    continue
                    
                content = row['content']
                metadata = row.get('metadata', {})
                
                # Factor 1: Penalización para TOCs y secciones de referencias
                if self.is_toc_section(content):
                    adjusted_scores[i] *= 0.3  # Penalización más fuerte (70% de reducción)
                    logger.debug(f"Chunk {chunk_id}: Penalizado como TOC: {adjusted_scores[i]:.4f}")
                    continue  # No aplicar otras bonificaciones a TOCs
                
                # Factor 2: Score de informatividad
                info_score = self.calculate_informative_score(content)
                
                # Escalado no lineal para enfatizar diferencias
                # Contenido muy informativo (>0.7) recibe boost, contenido poco informativo (<0.4) se penaliza
                if info_score > 0.7:
                    boost = 1.0 + (info_score - 0.7) * 0.8  # Hasta 24% de boost
                elif info_score < 0.4:
                    boost = 0.7 + (info_score * 0.75)  # Hasta 30% de penalización
                else:
                    boost = 1.0  # Neutral para valores medios
                
                adjusted_scores[i] *= boost
                logger.debug(f"Chunk {chunk_id}: Score informativo {info_score:.2f}, boost: {boost:.2f}")
                
                # Factor 3: Penalización para chunks muy cortos
                if len(content) < 200:
                    adjusted_scores[i] *= max(0.5, len(content) / 200)
                    logger.debug(f"Chunk {chunk_id}: Penalizado por ser corto: {len(content)} chars")
                
                # Factor 4: Boosting contextual basado en términos de la consulta
                if query_terms:
                    content_lower = content.lower()
                    # Contar cuántos términos de la consulta aparecen en el contenido
                    matching_terms = sum(1 for term in query_terms if term in content_lower)
                    
                    # Calcular ratio de coincidencia y frecuencia
                    term_ratio = matching_terms / len(query_terms) if query_terms else 0
                    
                    # Calcular frecuencia de términos (no solo presencia)
                    term_frequency = 0
                    for term in query_terms:
                        term_frequency += content_lower.count(term)
                    
                    # Normalizar frecuencia
                    term_freq_ratio = min(1.0, term_frequency / (len(content) * 0.01))
                    
                    # Boost combinando ratio y frecuencia
                    if term_ratio > 0 or term_freq_ratio > 0:
                        # Boost proporcional a coincidencia y frecuencia (hasta 60% de boost)
                        term_boost = 1.0 + min(0.6, (term_ratio * 0.4) + (term_freq_ratio * 0.2))
                        adjusted_scores[i] *= term_boost
                        logger.debug(f"Chunk {chunk_id}: Boost por términos: {term_boost:.2f} (match: {term_ratio:.2f}, freq: {term_freq_ratio:.2f})")
                
                # Limitar el boost/penalización para evitar valores extremos
                adjusted_scores[i] = max(0.1, min(1.5, adjusted_scores[i]))
        
        return adjusted_scores
    
    def _diversify_results(self, chunk_ids: List[str], similarities: List[float], 
                          doc_ids: List[str] = None, max_results: int = None) -> Tuple[List[str], List[float]]:
        """
        Diversifica resultados para evitar redundancia y sobrerrepresentación de documentos.
        
        Args:
            chunk_ids: Lista de IDs de chunks
            similarities: Lista de puntuaciones de similitud
            doc_ids: Lista opcional de IDs de documentos correspondientes a los chunks
            max_results: Número máximo de resultados a devolver
            
        Returns:
            Tuple de (chunk_ids diversificados, similitudes correspondientes)
        """
        if not chunk_ids or not similarities:
            return chunk_ids, similarities
            
        if max_results is None or max_results >= len(chunk_ids):
            return chunk_ids, similarities
            
        # Si no hay doc_ids, no podemos diversificar por documento
        if not doc_ids or len(doc_ids) != len(chunk_ids):
            # Simplemente tomar los top resultados
            return chunk_ids[:max_results], similarities[:max_results]
        
        # Crear lista de tuplas (chunk_id, similarity, doc_id)
        results = list(zip(chunk_ids, similarities, doc_ids))
        
        # Estrategia de diversificación:
        # 1. Tomar el mejor resultado primero
        # 2. Luego alternar entre documentos diferentes
        
        # Ordenar por similitud (descendente)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Obtener conjuntos de documentos
        unique_docs = set(doc_ids)
        
        # Estrategia: Tomar un chunk de cada documento en orden de relevancia
        selected = []
        seen_docs = set()
        
        # Primera ronda: tomar el mejor chunk de cada documento
        for chunk_id, similarity, doc_id in results:
            if doc_id not in seen_docs:
                selected.append((chunk_id, similarity, doc_id))
                seen_docs.add(doc_id)
                
                # Si ya tenemos suficientes resultados, terminar
                if len(selected) >= max_results:
                    break
        
        # Si todavía necesitamos más resultados, añadir segundos mejores chunks
        if len(selected) < max_results:
            for chunk_id, similarity, doc_id in results:
                if (chunk_id, similarity, doc_id) not in selected:
                    selected.append((chunk_id, similarity, doc_id))
                    
                    # Si ya tenemos suficientes resultados, terminar
                    if len(selected) >= max_results:
                        break
        
        # Extraer listas finales
        diverse_chunk_ids = [item[0] for item in selected]
        diverse_similarities = [item[1] for item in selected]
        
        logger.info(f"Diversificación: {len(chunk_ids)} chunks originales -> {len(diverse_chunk_ids)} diversos ({len(unique_docs)} docs únicos)")
        return diverse_chunk_ids, diverse_similarities
    
    async def search_faiss(
        self, 
        query_vector: np.ndarray, 
        k: int = None, 
        filter_dict: Dict = None, 
        query_text: str = None
    ) -> Dict[str, Any]:
        """
        Busca documentos similares usando FAISS.
        Obtiene resultados adicionales para filtraje posterior.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados (None para usar valor predeterminado)
            filter_dict: Filtros opcionales para la búsqueda
            query_text: Texto original de la consulta
            
        Returns:
            Diccionario con resultados y metadatos de la búsqueda
        """
        start_time = time.time()
        k = k or self.default_k
        
        # Solicitar más resultados para filtrar después (k*4)
        search_k = k * 4
        
        # Validar vector de consulta
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Asegurar shape correcto
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Realizar búsqueda en FAISS
        distances, indices = self.faiss_store.search(query_vector, k=search_k)
        
        # Convertir a lista para facilitar manipulación
        if isinstance(distances, np.ndarray):
            distances = distances.tolist()
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        # Calcular similitudes (inversas de distancias L2)
        max_distance = max([max(d) if d else 0 for d in distances]) if distances else 0
        if max_distance > 0:
            similarities = [[1 - (d / max_distance) for d in dist] for dist in distances]
        else:
            similarities = [[1.0 for _ in dist] for dist in distances]
        
        # Map FAISS indices to chunk_ids
        chunk_ids = []
        doc_ids = []
        if indices and len(indices) > 0 and len(indices[0]) > 0:
            faiss_indices = indices[0]
            chunk_ids, doc_ids = await self._map_faiss_indices_to_chunk_ids(faiss_indices)
        
        # Get adjusted scores
        if chunk_ids and similarities and len(similarities) > 0:
            adjusted_similarities = await self.adjust_relevance_scores(chunk_ids, similarities[0], query_text)
            
            # Diversificar resultados
            if doc_ids and len(doc_ids) == len(chunk_ids):
                chunk_ids, adjusted_similarities = self._diversify_results(
                    chunk_ids, adjusted_similarities, doc_ids, k
                )
            else:
                # Si no hay doc_ids, simplemente ordenar por puntuación
                chunks_with_scores = list(zip(chunk_ids, adjusted_similarities))
                chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar top k
                top_chunks = chunks_with_scores[:k]
                
                # Actualizar resultados
                chunk_ids = [c[0] for c in top_chunks]
                adjusted_similarities = [c[1] for c in top_chunks]
            
            # Actualizar similitudes
            similarities = [adjusted_similarities]
        
        search_time = time.time() - start_time
        
        return {
            'indices': indices,
            'distances': distances,
            'similarities': similarities,
            'chunk_ids': chunk_ids,
            'doc_ids': doc_ids[:len(chunk_ids)] if doc_ids else None,
            'search_time': search_time,
            'k': k,
            'mode': 'faiss'
        }
    
    async def search_pgvector(
        self, 
        query_vector: np.ndarray, 
        k: int = None, 
        similarity_threshold: float = None,
        model_name: str = None,
        query_text: str = None
    ) -> Dict[str, Any]:
        """
        Busca documentos similares usando pgvector.
        Obtiene resultados adicionales para filtrado y diversificación.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados
            similarity_threshold: Umbral de similitud mínima
            model_name: Filtro opcional por modelo de embedding
            query_text: Texto original de la consulta
            
        Returns:
            Diccionario con resultados y metadatos de la búsqueda
        """
        start_time = time.time()
        k = k or self.default_k
        similarity_threshold = similarity_threshold or self.min_similarity
        
        # Solicitar más resultados para filtrar después
        search_k = k * 3
        
        # Realizar búsqueda en pgvector
        try:
            results = await self.embedding_repository.search_similar(
                query_vector=query_vector,
                k=search_k,
                distance_threshold=similarity_threshold,
                model_name=model_name
            )
            
            # Extraer resultados
            chunk_ids = [r['chunk_id'] for r in results]
            similarities = [r['similarity'] for r in results]
            doc_ids = [r.get('doc_id') for r in results] if all('doc_id' in r for r in results) else None
            
            # Ajustar puntuaciones basadas en calidad del contenido
            adjusted_similarities = await self.adjust_relevance_scores(chunk_ids, similarities, query_text)
            
            # Diversificar resultados
            if doc_ids and len(doc_ids) == len(chunk_ids):
                chunk_ids, adjusted_similarities = self._diversify_results(
                    chunk_ids, adjusted_similarities, doc_ids, k
                )
            else:
                # Si no hay doc_ids, simplemente ordenar por puntuación
                chunks_with_scores = list(zip(chunk_ids, adjusted_similarities))
                chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar top k
                top_chunks = chunks_with_scores[:k]
                
                # Actualizar resultados
                chunk_ids = [c[0] for c in top_chunks]
                adjusted_similarities = [c[1] for c in top_chunks]
            
            # Actualizar resultados
            filtered_results = [r for r in results if r['chunk_id'] in chunk_ids]
            
            search_time = time.time() - start_time
            
            return {
                'results': filtered_results[:k],
                'chunk_ids': chunk_ids,
                'doc_ids': doc_ids[:len(chunk_ids)] if doc_ids else None,
                'similarities': [adjusted_similarities],  # Formato compatible con FAISS
                'search_time': search_time,
                'k': k,
                'mode': 'pgvector'
            }
        except Exception as e:
            logger.error(f"Error en búsqueda pgvector: {e}")
            return {
                'results': [],
                'chunk_ids': [],
                'doc_ids': [],
                'similarities': [[]],
                'search_time': time.time() - start_time,
                'k': k,
                'mode': 'pgvector',
                'error': str(e)
            }
    
    async def hybrid_search(
        self, 
        query_vector: np.ndarray, 
        k: int = None, 
        model_name: str = None,
        query_text: str = None
    ) -> Dict[str, Any]:
        """
        Realiza una búsqueda híbrida combinando FAISS y pgvector.
        Incluye mecanismos de diversificación y ranking mejorado.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados
            model_name: Filtro opcional por modelo de embedding
            query_text: Texto original de la consulta
            
        Returns:
            Diccionario con resultados combinados
        """
        start_time = time.time()
        k = k or self.default_k
        
        # Solicitar más resultados para filtrar después
        search_k = k * 3
        
        # Realizar búsquedas en paralelo
        faiss_task = asyncio.create_task(self.search_faiss(query_vector, k=search_k, query_text=query_text))
        pgvector_task = asyncio.create_task(
            self.search_pgvector(query_vector, k=search_k, model_name=model_name, query_text=query_text)
        )
        
        # Esperar resultados
        faiss_results, pgvector_results = await asyncio.gather(faiss_task, pgvector_task)
        
        # Verificar si alguna búsqueda falló
        if not faiss_results.get('chunk_ids') and not pgvector_results.get('chunk_ids'):
            logger.warning("Ambas búsquedas (FAISS y pgvector) fallaron o no retornaron resultados")
            return {
                'results': [],
                'chunk_ids': [],
                'doc_ids': [],
                'search_time': time.time() - start_time,
                'k': k,
                'mode': 'hybrid_failed'
            }
        
        # Si una búsqueda falló, usar la otra
        if not faiss_results.get('chunk_ids'):
            logger.warning("Búsqueda FAISS falló, usando solo pgvector")
            return pgvector_results
        
        if not pgvector_results.get('chunk_ids'):
            logger.warning("Búsqueda pgvector falló, usando solo FAISS")
            return faiss_results
        
        # Combinar resultados
        faiss_chunk_ids = faiss_results.get('chunk_ids', [])
        faiss_similarities = faiss_results['similarities'][0] if faiss_results['similarities'] else []
        faiss_doc_ids = faiss_results.get('doc_ids', [])
        
        pgvector_chunk_ids = pgvector_results.get('chunk_ids', [])
        pgvector_similarities = pgvector_results.get('similarities', [[]])[0]
        pgvector_doc_ids = pgvector_results.get('doc_ids', [])
        
        # Unir resultados con diccionario para eliminar duplicados
        # y mantener la mayor similitud para cada chunk_id
        combined_results = {}
        combined_doc_ids = {}
        
        for i, chunk_id in enumerate(faiss_chunk_ids):
            if chunk_id and i < len(faiss_similarities):
                combined_results[chunk_id] = {
                    'similarity': faiss_similarities[i],
                    'source': 'faiss'
                }
                # Guardar doc_id si está disponible
                if faiss_doc_ids and i < len(faiss_doc_ids):
                    combined_doc_ids[chunk_id] = faiss_doc_ids[i]
        
        for i, chunk_id in enumerate(pgvector_chunk_ids):
            if chunk_id and i < len(pgvector_similarities):
                if chunk_id not in combined_results or pgvector_similarities[i] > combined_results[chunk_id]['similarity']:
                    combined_results[chunk_id] = {
                        'similarity': pgvector_similarities[i],
                        'source': 'pgvector'
                    }
                    # Guardar doc_id si está disponible
                    if pgvector_doc_ids and i < len(pgvector_doc_ids):
                        combined_doc_ids[chunk_id] = pgvector_doc_ids[i]
        
        # Obtener chunk_ids y similitudes únicos
        unique_chunk_ids = list(combined_results.keys())
        unique_similarities = [data['similarity'] for data in combined_results.values()]
        unique_sources = [data['source'] for data in combined_results.values()]
        
        # Obtener doc_ids para diversificación
        if combined_doc_ids:
            unique_doc_ids = [combined_doc_ids.get(chunk_id) for chunk_id in unique_chunk_ids]
        else:
            unique_doc_ids = None
        
        # Ajustar puntuaciones de relevancia basadas en calidad del contenido
        adjusted_similarities = await self.adjust_relevance_scores(unique_chunk_ids, unique_similarities, query_text)
        
        # Diversificar resultados
        if unique_doc_ids:
            final_chunk_ids, final_similarities = self._diversify_results(
                unique_chunk_ids, adjusted_similarities, unique_doc_ids, k
            )
            # Recalcular fuentes para los resultados diversificados
            final_sources = []
            for chunk_id in final_chunk_ids:
                idx = unique_chunk_ids.index(chunk_id)
                final_sources.append(unique_sources[idx])
        else:
            # Re-ordenar con puntuaciones ajustadas y limitar a k
            combined_with_adjusted = list(zip(unique_chunk_ids, adjusted_similarities, unique_sources))
            combined_with_adjusted.sort(key=lambda x: x[1], reverse=True)
            top_results = combined_with_adjusted[:k]
            
            # Extraer listas finales
            final_chunk_ids = [item[0] for item in top_results]
            final_similarities = [item[1] for item in top_results]
            final_sources = [item[2] for item in top_results]
        
        # Extraer doc_ids finales
        final_doc_ids = [combined_doc_ids.get(chunk_id) for chunk_id in final_chunk_ids] if combined_doc_ids else None
        
        hybrid_time = time.time() - start_time
        
        return {
            'chunk_ids': final_chunk_ids,
            'doc_ids': final_doc_ids,
            'similarities': [final_similarities],
            'sources': final_sources,
            'search_time': hybrid_time,
            'faiss_time': faiss_results.get('search_time', 0),
            'pgvector_time': pgvector_results.get('search_time', 0),
            'k': k,
            'mode': 'hybrid'
        }
    
    async def _map_faiss_indices_to_chunk_ids(self, indices: List[int]) -> Tuple[List[str], List[str]]:
        """
        Convierte índices de FAISS a chunk_ids y doc_ids usando la base de datos.
        
        Args:
            indices: Lista de índices FAISS
            
        Returns:
            Tuple de (Lista de chunk_ids, Lista de doc_ids)
        """
        if not indices:
            return [], []
        
        try:
            pool = await self.embedding_repository._get_pool()
            chunk_ids = []
            doc_ids = []
            
            async with pool.acquire() as conn:
                for idx in indices:
                    # Buscar el embedding con ese faiss_index_id
                    row = await conn.fetchrow("""
                        SELECT e.chunk_id, c.doc_id
                        FROM embeddings e
                        LEFT JOIN chunks c ON e.chunk_id = c.chunk_id
                        WHERE e.faiss_index_id = $1
                    """, idx)
                    
                    if row:
                        chunk_ids.append(row['chunk_id'])
                        doc_ids.append(row['doc_id'])
                    else:
                        chunk_ids.append(None)
                        doc_ids.append(None)
                        logger.warning(f"No se encontró chunk_id para índice FAISS {idx}")
            
            return chunk_ids, doc_ids
        
        except Exception as e:
            logger.error(f"Error mapeando índices FAISS a chunk_ids: {e}")
            return [None] * len(indices), [None] * len(indices)
    
    async def retrieve(
        self, 
        query_vector: np.ndarray, 
        k: int = None, 
        mode: str = None,
        model_name: str = None,
        query_text: str = None
    ) -> Dict[str, Any]:
        """
        Realiza una búsqueda de documentos según el modo especificado.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados
            mode: Modo de búsqueda ('faiss', 'pgvector', 'hybrid')
            model_name: Filtro opcional por modelo de embedding
            query_text: Texto original de la consulta
            
        Returns:
            Resultados de la búsqueda
        """
        k = k or self.default_k
        mode = mode or self.search_mode
        
        if mode == 'faiss':
            return await self.search_faiss(query_vector, k, query_text=query_text)
        elif mode == 'pgvector':
            return await self.search_pgvector(query_vector, k, model_name=model_name, query_text=query_text)
        elif mode == 'hybrid':
            return await self.hybrid_search(query_vector, k, model_name=model_name, query_text=query_text)
        else:
            raise ValueError(f"Modo de búsqueda no válido: {mode}")
    
    async def retrieve_by_text(
        self,
        query_processor,
        query_text: str,
        k: int = None,
        mode: str = None,
        model_key: str = None
    ) -> Dict[str, Any]:
        """
        Realiza una búsqueda a partir de texto directamente.
        
        Args:
            query_processor: Procesador de consultas para generar el embedding
            query_text: Texto de la consulta
            k: Número de resultados
            mode: Modo de búsqueda
            model_key: Clave del modelo para generar embeddings
            
        Returns:
            Resultados de la búsqueda
        """
        # Procesar consulta
        query_data = await query_processor.process_query(query_text, model_key)
        
        # Realizar búsqueda
        results = await self.retrieve(
            query_vector=query_data['embedding'],
            k=k,
            mode=mode,
            model_name=query_data['embedding_model'],
            query_text=query_text  # Pass the original query for context-aware boosting
        )
        
        # Añadir información de la consulta
        results['query'] = {
            'text': query_text,
            'processed': query_data['processed_query'],
            'language': query_data['language'],
            'type': query_data['query_type']
        }
        
        return results