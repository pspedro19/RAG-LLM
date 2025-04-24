# app/query/query_processor.py

import logging
import re
import unicodedata
import numpy as np
from typing import Dict, Any, List, Optional, Union, Set
import time
import string

# Importar módulos para traducción
import argostranslate.package
import argostranslate.translate

from app.core.config.config import Config
from app.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Procesa consultas en lenguaje natural para su uso en sistemas RAG.
    Maneja normalización, preprocesamiento, expansión, traducción y conversión a embeddings.
    """
    
    def __init__(self, config: Config, embedding_service: EmbeddingService):
        """
        Inicializa el procesador de consultas.
        
        Args:
            config: Configuración del sistema
            embedding_service: Servicio para generación de embeddings
        """
        self.config = config
        self.embedding_service = embedding_service
        
        # Configuración de preprocesamiento
        self.min_query_length = 3
        self.max_query_length = 512
        self.default_model = 'miniLM'
        
        # Flag para expansión de consulta
        self.enable_query_expansion = getattr(config, 'ENABLE_QUERY_EXPANSION', False)
        
        # Flag para búsqueda cruzada entre idiomas
        self.enable_cross_lingual = getattr(config, 'ENABLE_CROSS_LINGUAL', True)
        
        # Inicializar traducción si está habilitada
        if self.enable_cross_lingual:
            self.install_translation_models()
        
        # Expresiones regulares para limpieza
        self.cleanup_patterns = [
            (re.compile(r'\s+'), ' '),  # Múltiples espacios a uno solo
            (re.compile(r'[^\w\s\?\.\,\!\:\;\-]'), '')  # Elimina caracteres especiales excepto puntuación común
        ]
        
        # Stopwords - palabras comunes que no aportan significado semántico importante
        # Lista genérica que funciona para diferentes dominios
        self.stopwords = {
            'en': {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'as', 'at', 'by', 'for',
                   'in', 'to', 'with', 'about', 'against', 'between', 'into', 'through',
                   'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
                   'on', 'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there',
                   'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'more',
                   'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                   'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
                   'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                   'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'might',
                   'must', 'shall', 'may'},
            'es': {'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con',
                   'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e',
                   'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran',
                   'eras', 'eres', 'es', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba',
                   'estabais', 'estaban', 'estabas', 'estad', 'estada', 'estadas', 'estado',
                   'estados', 'estamos', 'estando', 'estar', 'estaremos', 'estará', 'estarán',
                   'estarás', 'estaré', 'estaréis', 'estaría', 'estaríais', 'estaríamos',
                   'estarían', 'estarías', 'estas', 'este', 'estemos', 'esto', 'estos', 'estoy',
                   'estuve', 'estuviera', 'estuvierais', 'estuvieran', 'estuvieras',
                   'estuvieron', 'estuviese', 'estuvieseis', 'estuviesen', 'estuvieses',
                   'estuvimos', 'estuviste', 'estuvisteis', 'estuviéramos', 'estuviésemos',
                   'estuvo', 'hay', 'ha', 'habéis', 'haber', 'habida', 'habidas', 'habido',
                   'habidos', 'habiendo', 'habremos', 'habrá', 'habrán', 'habrás', 'habré',
                   'habréis', 'habría', 'habríais', 'habríamos', 'habrían', 'habrías', 'han',
                   'has', 'hasta', 'hay', 'haya', 'hayamos', 'hayan', 'hayas', 'hayáis', 'he',
                   'hemos', 'hube', 'hubiera', 'hubierais', 'hubieran', 'hubieras', 'hubieron',
                   'hubiese', 'hubieseis', 'hubiesen', 'hubieses', 'hubimos', 'hubiste',
                   'hubisteis', 'hubiéramos', 'hubiésemos', 'hubo', 'la', 'las', 'le', 'les',
                   'lo', 'los', 'me', 'mi', 'mis', 'mucho', 'muchos', 'muy', 'más', 'mí', 'mía',
                   'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos', 'nosotras', 'nosotros',
                   'nuestra', 'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras',
                   'otro', 'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien',
                   'quienes', 'qué', 'se', 'sea', 'seamos', 'sean', 'seas', 'seremos', 'será',
                   'serán', 'serás', 'seré', 'seréis', 'sería', 'seríais', 'seríamos', 'serían',
                   'serías', 'seáis', 'si', 'sido', 'siendo', 'sin', 'sobre', 'sois', 'somos',
                   'son', 'soy', 'su', 'sus', 'suya', 'suyas', 'suyo', 'suyos', 'sí', 'también',
                   'tanto', 'te', 'tendremos', 'tendrá', 'tendrán', 'tendrás', 'tendré',
                   'tendréis', 'tendría', 'tendríais', 'tendríamos', 'tendrían', 'tendrías',
                   'tened', 'tenemos', 'tenga', 'tengamos', 'tengan', 'tengas', 'tengo',
                   'tengáis', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis',
                   'tenía', 'teníais', 'teníamos', 'tenían', 'tenías', 'ti', 'tiene', 'tienen',
                   'tienes', 'todo', 'todos', 'tu', 'tus', 'tuve', 'tuviera', 'tuvierais',
                   'tuvieran', 'tuvieras', 'tuvieron', 'tuviese', 'tuvieseis', 'tuviesen',
                   'tuvieses', 'tuvimos', 'tuviste', 'tuvisteis', 'tuviéramos', 'tuviésemos',
                   'tuvo', 'tuya', 'tuyas', 'tuyo', 'tuyos', 'tú', 'un', 'una', 'uno', 'unos',
                   'vosotras', 'vosotros', 'vuestra', 'vuestras', 'vuestro', 'vuestros', 'y',
                   'ya', 'yo', 'él', 'éramos'}
        }
    
    def install_translation_models(self):
        """Instala modelos de traducción si no están presentes"""
        try:
            # Verificar si se necesita descargar paquetes
            if not argostranslate.package.get_installed_packages():
                logger.info("Descargando e instalando modelos de traducción...")
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()
                
                # Buscar paquetes es-en y en-es
                for package in available_packages:
                    if ((package.from_code == "es" and package.to_code == "en") or
                        (package.from_code == "en" and package.to_code == "es")):
                        logger.info(f"Instalando modelo de traducción: {package.from_code} → {package.to_code}")
                        argostranslate.package.install_from_path(package.download())
                
                logger.info("Modelos de traducción instalados correctamente")
            else:
                logger.info("Modelos de traducción ya instalados")
        except Exception as e:
            logger.error(f"Error al instalar modelos de traducción: {e}")
            logger.warning("La búsqueda cruzada entre idiomas podría no funcionar correctamente")
    
    def translate_query(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Traduce la consulta al idioma objetivo.
        
        Args:
            query: Consulta a traducir
            source_lang: Código de idioma origen (en, es)
            target_lang: Código de idioma destino (en, es)
            
        Returns:
            Consulta traducida
        """
        if source_lang == target_lang:
            return query
            
        try:
            # Asegurarnos de que los códigos de idioma sean correctos
            source_lang = source_lang.lower()
            target_lang = target_lang.lower()
            
            # Solo soportamos traducción entre inglés y español
            if not ((source_lang == 'en' and target_lang == 'es') or 
                    (source_lang == 'es' and target_lang == 'en')):
                logger.warning(f"Traducción no soportada: {source_lang} → {target_lang}")
                return query
                
            # Realizar traducción
            translated = argostranslate.translate.translate(query, source_lang, target_lang)
            logger.debug(f"Traducción: '{query}' ({source_lang}) → '{translated}' ({target_lang})")
            return translated
        except Exception as e:
            logger.error(f"Error en traducción: {e}")
            return query  # Fallback a consulta original
    
    def normalize_query(self, query: str) -> str:
        """
        Normaliza una consulta: elimina caracteres especiales, normaliza espacios, etc.
        
        Args:
            query: Consulta original
            
        Returns:
            Consulta normalizada
        """
        # Convertir a texto si no lo es
        if not isinstance(query, str):
            query = str(query)
        
        # Normalización Unicode (NFKD) para manejar acentos y caracteres especiales
        query = unicodedata.normalize('NFKD', query)
        
        # Aplicar patrones de limpieza
        for pattern, replacement in self.cleanup_patterns:
            query = pattern.sub(replacement, query)
        
        # Normalizar espacios
        query = query.strip()
        
        return query
    
    def extract_keywords(self, text: str, language: str = 'en') -> List[str]:
        """
        Extrae palabras clave de un texto (elimina stopwords y palabras cortas).
        Esta función es agnóstica al dominio y funciona con cualquier tipo de consulta.
        
        Args:
            text: Texto del cual extraer palabras clave
            language: Idioma del texto ('en' o 'es')
            
        Returns:
            Lista de palabras clave
        """
        # Normalizar y dividir en palabras
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Obtener stopwords para el idioma
        stops = self.stopwords.get(language, self.stopwords['en'])
        
        # Filtrar palabras cortas y stopwords
        keywords = [word for word in words if len(word) > 2 and word not in stops]
        
        return keywords
    
    def expand_query(self, query: str, language: str = 'en') -> str:
        """
        Expande una consulta con sinónimos o términos relacionados (implementación simple).
        
        Args:
            query: Consulta original
            language: Idioma de la consulta
            
        Returns:
            Consulta expandida
        """
        if not self.enable_query_expansion:
            return query
            
        # Extraer palabras clave
        keywords = self.extract_keywords(query, language)
        
        # Aquí implementaríamos la expansión con sinónimos o términos relacionados
        # Por ejemplo, usando WordNet o un diccionario de sinónimos
        # En esta implementación simple, solo añadimos algunas variaciones básicas
        
        expanded_terms = []
        
        # Aquí se podrían añadir reglas específicas de expansión
        # En esta implementación simple, añadimos algunas variaciones comunes
        for keyword in keywords:
            expanded_terms.append(keyword)
            
            # Añadir forma singular/plural básica (muy simplificado)
            if language == 'en':
                if keyword.endswith('s') and len(keyword) > 4:
                    expanded_terms.append(keyword[:-1])  # Quitar 's'
                else:
                    expanded_terms.append(keyword + 's')  # Añadir 's'
            elif language == 'es':
                if keyword.endswith('es') and len(keyword) > 4:
                    expanded_terms.append(keyword[:-2])  # Quitar 'es'
                elif keyword.endswith('s') and len(keyword) > 4:
                    expanded_terms.append(keyword[:-1])  # Quitar 's'
        
        # Unir consulta original con términos expandidos únicos
        expanded_query = query
        
        # Añadir términos nuevos que no estén ya en la consulta
        new_terms = ' '.join([term for term in expanded_terms if term.lower() not in query.lower()])
        
        if new_terms:
            expanded_query = f"{query} {new_terms}"
            logger.debug(f"Consulta expandida: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocesa una consulta: normalización, validación, detección de idioma, etc.
        
        Args:
            query: Consulta original
            
        Returns:
            Diccionario con consulta procesada y metadatos
        """
        start_time = time.time()
        
        # Normalizar consulta
        normalized_query = self.normalize_query(query)
        
        # Validar longitud
        if len(normalized_query) < self.min_query_length:
            logger.warning(f"Consulta demasiado corta: '{normalized_query}'")
            normalized_query = query  # Usar original si la normalizada es muy corta
        
        if len(normalized_query) > self.max_query_length:
            logger.warning(f"Consulta truncada de {len(normalized_query)} a {self.max_query_length} caracteres")
            normalized_query = normalized_query[:self.max_query_length]
        
        # Detectar idioma (implementación mejorada)
        language = self._detect_language(normalized_query)
        
        # Detectar tipo de consulta (QA, búsqueda, etc.)
        query_type = self._detect_query_type(normalized_query)
        
        # Extraer palabras clave
        keywords = self.extract_keywords(normalized_query, language)
        
        # Expandir consulta si está habilitado
        if self.enable_query_expansion:
            expanded_query = self.expand_query(normalized_query, language)
        else:
            expanded_query = normalized_query
        
        # Preparar traducción cruzada si está habilitada
        if self.enable_cross_lingual:
            # Determinar idioma alterno (para búsqueda cruzada)
            alternate_language = 'en' if language == 'es' else 'es'
            
            # Traducir consulta al idioma alterno
            translated_query = self.translate_query(normalized_query, language, alternate_language)
            
            # Extraer palabras clave de la traducción
            translated_keywords = self.extract_keywords(translated_query, alternate_language)
        else:
            translated_query = None
            alternate_language = None
            translated_keywords = []
        
        processing_time = time.time() - start_time
        
        result = {
            'original_query': query,
            'processed_query': normalized_query,
            'expanded_query': expanded_query,
            'language': language,
            'query_type': query_type,
            'token_count': len(normalized_query.split()),
            'keywords': keywords,
            'processing_time': processing_time
        }
        
        # Añadir información de traducción si está habilitada
        if self.enable_cross_lingual and translated_query:
            result.update({
                'translated_query': translated_query,
                'alternate_language': alternate_language,
                'translated_keywords': translated_keywords
            })
        
        return result
    
    def _detect_language(self, query: str) -> str:
        """
        Detecta el idioma de la consulta (implementación mejorada).
        
        Args:
            query: Consulta normalizada
            
        Returns:
            Código de idioma (es, en, etc.)
        """
        # Esta es una implementación básica que podría reemplazarse por una librería como langdetect
        # Para sistemas en producción, considerar usar una librería especializada
        
        query_lower = query.lower()
        
        # Palabras y caracteres específicos de cada idioma
        spanish_chars = set('áéíóúüñ¿¡')
        
        # Detectores específicos de cada idioma
        spanish_markers = {
            'palabras': ['qué', 'cómo', 'dónde', 'cuándo', 'quién', 'cuál', 'cuáles', 'por qué', 
                       'porque', 'según', 'entre', 'hasta', 'desde', 'durante', 'mediante'],
            'articulos': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'del', 'al'],
            'verbos': ['es', 'está', 'son', 'están', 'ha', 'han', 'fue', 'fueron', 'tengo', 'tiene', 'quiero']
        }
        
        english_markers = {
            'words': ['what', 'how', 'where', 'when', 'who', 'which', 'whom', 'whose', 'why',
                     'because', 'through', 'though', 'although', 'whether', 'during'],
            'articles': ['the', 'a', 'an', 'some', 'any'],
            'verbs': ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'want', 'need']
        }
        
        # Contar coincidencias por categoría
        spanish_count = 0
        english_count = 0
        
        # Buscar caracteres españoles específicos
        if any(c in spanish_chars for c in query_lower):
            spanish_count += 3  # Alta ponderación para caracteres específicos
        
        # Comprobar marcadores por categoría
        for category, words in spanish_markers.items():
            spanish_count += sum(1 for word in words if f" {word} " in f" {query_lower} ")
            
        for category, words in english_markers.items():
            english_count += sum(1 for word in words if f" {word} " in f" {query_lower} ")
        
        # Heurística adicional: distribución de letras
        vowel_ratio = sum(1 for c in query_lower if c in 'aeiou') / len(query_lower) if query_lower else 0
        if vowel_ratio > 0.45:  # El español tiende a tener más vocales
            spanish_count += 1
        if 'th' in query_lower or 'wh' in query_lower:  # Combinaciones más comunes en inglés
            english_count += 1
            
        logger.debug(f"Detección de idioma: español={spanish_count}, inglés={english_count}")
        
        # Umbral mínimo para evitar falsos positivos en textos cortos o ambiguos
        if max(spanish_count, english_count) < 1 and len(query_lower) < 10:
            return 'en'  # Por defecto inglés para consultas muy cortas
            
        return 'es' if spanish_count > english_count else 'en'
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detecta el tipo de consulta (pregunta, búsqueda, etc.).
        Mejorado para detectar más patrones y tipos de consulta.
        
        Args:
            query: Consulta normalizada
            
        Returns:
            Tipo de consulta
        """
        query_lower = query.lower()
        
        # Verificar si es una pregunta
        if '?' in query or re.search(r'^(qué|cómo|dónde|cuándo|quién|cuál|por qué|what|how|where|when|who|which|why)\b', query_lower):
            
            # Subtipos de preguntas
            if re.search(r'\b(diferencia|comparar|versus|vs|difference|compare)\b', query_lower):
                return 'comparison'
            elif re.search(r'\b(definir|definición|qué es|what is|meaning|significado)\b', query_lower):
                return 'definition'
            elif re.search(r'\b(cómo|how to|manera|forma|steps|pasos)\b', query_lower):
                return 'how_to'
            else:
                return 'question'
        
        # Verificar si es una búsqueda específica
        if any(token in query_lower for token in ['buscar', 'encontrar', 'search', 'find', 'lookup']):
            return 'search'
            
        # Verificar si es un comando o instrucción
        if re.match(r'^(mostrar|listar|enumerar|show|list|give me|tell me|provide|muestra|dame|dime|proporciona)', query_lower):
            return 'command'
            
        # Verificar si es una comparación
        if re.search(r'\b(vs|versus|compared to|comparado con|diferencia entre|difference between)\b', query_lower):
            return 'comparison'
            
        # Detectar pedidos de resumen o explicación
        if re.search(r'\b(resume|resumir|resumen|summary|summarize|explain|explicar|explica)\b', query_lower):
            return 'summary'
        
        # Por defecto, considerar como búsqueda general
        return 'general'
    
    def generate_embedding(self, query: str, model_key: str = None) -> np.ndarray:
        """
        Genera un embedding para la consulta.
        
        Args:
            query: Consulta a vectorizar
            model_key: Clave del modelo a utilizar (None para usar el predeterminado)
            
        Returns:
            Vector de embedding
        """
        # Procesar consulta
        processed = self.preprocess_query(query)
        
        # Determinar qué texto usar para el embedding
        if self.enable_query_expansion and 'expanded_query' in processed:
            query_text = processed['expanded_query']
        else:
            query_text = processed['processed_query']
        
        # Usar modelo predeterminado si no se especifica
        model_key = model_key or self.default_model
        
        # Generar embedding
        start_time = time.time()
        embedding = self.embedding_service.generate_embeddings([query_text], model_key)
        
        generation_time = time.time() - start_time
        logger.debug(f"Embedding generado en {generation_time:.3f}s")
        
        return embedding[0]
    
    async def process_query(self, query: str, model_key: str = None) -> Dict[str, Any]:
        """
        Procesa una consulta completa: preprocesamiento + generación de embedding.
        Incluye traducción si está habilitada la búsqueda cruzada entre idiomas.
        
        Args:
            query: Consulta original
            model_key: Clave del modelo a utilizar
            
        Returns:
            Diccionario con consulta procesada, metadatos y embedding
        """
        # Preprocesar consulta
        processed = self.preprocess_query(query)
        
        # Determinar qué texto usar para el embedding
        if self.enable_query_expansion and 'expanded_query' in processed:
            query_for_embedding = processed['expanded_query']
        else:
            query_for_embedding = processed['processed_query']
        
        # Generar embedding
        embedding = self.generate_embedding(query_for_embedding, model_key)
        
        # Añadir embedding al resultado
        processed['embedding'] = embedding
        processed['embedding_model'] = model_key or self.default_model
        processed['embedding_dimension'] = len(embedding)
        processed['embedding_source'] = 'expanded_query' if 'expanded_query' in processed and self.enable_query_expansion else 'processed_query'
        
        return processed