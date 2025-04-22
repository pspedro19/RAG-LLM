# app2/ingestion/document_processor.py

import logging
import uuid
import os
from typing import List, Dict, Any, Optional, Union
import tiktoken
import re
import hashlib

from app2.core.config.config import Config
from app2.ingestion.document_reader import DocumentReader

logger = logging.getLogger(__name__)

# Clase para entrada de documentos
class DocumentInput:
    """Estructura para entrada de documentos."""
    def __init__(self, title: str, content: str, metadata: dict = None, model_name: str = "miniLM"):
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.model_name = model_name

class DocumentProcessor:
    """
    Procesa documentos para dividirlos en chunks para indexación y recuperación.
    """
    
    def __init__(self, config: Config, chunk_repository=None):
        """
        Inicializa el procesador de documentos.
        
        Args:
            config: Configuración del sistema
            chunk_repository: Repositorio opcional para almacenar chunks directamente
        """
        self.config = config
        self.chunk_repository = chunk_repository
        
        # Inicializar tokenizador según la configuración
        tokenizer_name = getattr(config, 'TOKENIZER', "cl100k_base")
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            logger.warning(f"Error cargando tokenizador {tokenizer_name}: {e}. Usando cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configuración para chunking
        self.chunk_size = getattr(config, 'CHUNK_SIZE', 1000)
        self.chunk_overlap = getattr(config, 'CHUNK_OVERLAP', 200)
        
        # Compilar expresiones regulares para limpieza de texto
        self.cleanup_patterns = [
            (re.compile(r'\s+'), ' '),  # Múltiples espacios a uno solo
            (re.compile(r'^\s+|\s+$', re.MULTILINE), ''),  # Espacios al inicio/fin de líneas
        ]
        
        # Compilar patrones para reestructuración de texto
        self.restructure_patterns = [
            # Recomponer palabras cortadas por saltos de línea
            (re.compile(r'(\w)-\s*\n\s*(\w)'), r'\1\2'),
            # Preservar saltos de párrafo pero unir líneas dentro del mismo párrafo
            (re.compile(r'(\S)\n(?!\s*\n)(\S)'), r'\1 \2'),
            # Eliminar caracteres de control excepto saltos de línea
            (re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'), '')
        ]
        
        # Add TOC detection patterns
        self.toc_patterns = [
            re.compile(r'(?i)^(table of contents|contents|índice|índice de contenidos)[\s\n]*$'),
            re.compile(r'(?i)^(list of (tables|figures)|lista de (tablas|figuras))[\s\n]*$'),
            re.compile(r'(?:Figure|Table|Figura|Tabla)\s+\d+[\.\:]\s+'),
            re.compile(r'\d+\s*\.\s*\d+\s*(?:\.\s*\d+)?\s+[A-Z]')  # Numbered section headings
        ]
        
        # Patrones para detectar divisiones naturales para chunking
        self.natural_breaks = [
            # Encabezados principales
            re.compile(r'\n\s*#{1,3}\s+[^\n]+\n'),
            # Encabezados de sección numerados (ej. "2.1 Introducción")
            re.compile(r'\n\s*\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n]+\n'),
            # Divisiones por capítulos y secciones
            re.compile(r'\n\s*(?:Chapter|Section|Capítulo|Sección)\s+\d+[^\n]*\n'),
            # Fin de tabla
            re.compile(r'\n\s*\|\s*-{3,}\s*\|\s*\n'),
            # Fin de lista
            re.compile(r'\n\s*\n(?!\s*[-*•])')
        ]
        
        # Patrones para detectar estructuras a preservar (no cortar a media tabla, etc.)
        self.preserve_structures = [
            # Tablas markdown/ASCII
            (re.compile(r'\|\s*[-:]+\s*\|'), re.compile(r'\n\s*\n')),
            # Bloques de código
            (re.compile(r'```\w*\n'), re.compile(r'```\s*\n')),
            # Listas con viñetas
            (re.compile(r'(?:\n\s*[-*•]\s+[^\n]+)+'), None)
        ]
    
    def validate_document(self, doc: Union[Dict, DocumentInput]) -> DocumentInput:
        """
        Valida un documento de entrada y lo convierte a DocumentInput.
        
        Args:
            doc: Documento a validar (diccionario o DocumentInput)
            
        Returns:
            DocumentInput validado
        """
        if isinstance(doc, dict):
            doc = DocumentInput(**doc)
        
        if not doc.content.strip():
            raise ValueError("El contenido del documento no puede estar vacío.")
        
        return doc
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto para mejorar la calidad del chunking.
        Mejora la limpieza para evitar palabras cortadas y preservar estructura.
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Texto preprocesado
        """
        # Verificar si text es un diccionario (error detectado)
        if isinstance(text, dict) and 'content' in text:
            text = text['content']
        elif not isinstance(text, str):
            # Intentar convertir a string como último recurso
            text = str(text)
        
        # Normalizar saltos de línea
        text = text.replace('\r\n', '\n')
        
        # Aplicar patrones de reestructuración primero
        for pattern, replacement in self.restructure_patterns:
            text = pattern.sub(replacement, text)
        
        # Aplicar patrones de limpieza
        for pattern, replacement in self.cleanup_patterns:
            text = pattern.sub(replacement, text)
        
        # Eliminar múltiples saltos de línea consecutivos (preservar párrafos)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Eliminar caracteres Unicode no válidos o problemáticos
        text = ''.join(ch for ch in text if ord(ch) < 65536)
        
        return text.strip()
    
    def is_toc_section(self, text: str) -> bool:
        """
        Detecta si un texto parece ser una tabla de contenido o lista de figuras/tablas.
        Mejorado para distinguir mejor tablas de datos vs. TOCs.
        
        Args:
            text: Texto a analizar
            
        Returns:
            True si parece una tabla de contenido, False en caso contrario
        """
        # Check for TOC patterns
        toc_indicators = 0
        
        # Check patterns
        for pattern in self.toc_patterns:
            if pattern.search(text):
                toc_indicators += 1
                
        # Count dots, page numbers and other TOC indicators
        lines = text.split('\n')
        total_lines = len(lines)
        dots_count = text.count('.')
        numbers_count = sum(c.isdigit() for c in text)
        
        # Calculate ratio of numbers and dots to text length
        dots_numbers_ratio = (dots_count + numbers_count) / len(text) if len(text) > 0 else 0
        
        # Count lines with page number patterns at the end
        page_number_lines = 0
        for line in lines:
            # Patrones más estrictos para números de página (típico en TOCs)
            if re.search(r'\.\.\.\.*\d+$', line) or re.search(r'\s{3,}\d+$', line):
                page_number_lines += 1
        
        page_number_ratio = page_number_lines / total_lines if total_lines > 0 else 0
        
        # Verificar si hay muchas entradas "Figure X" o "Table X" seguidas
        figure_table_count = len(re.findall(r'(Figure|Table|Figura|Tabla)\s+\d+', text))
        figure_table_ratio = figure_table_count / total_lines if total_lines > 0 else 0
        
        # Excepción para tablas de datos que no son TOC
        if re.search(r'Table \d+:', text) and dots_numbers_ratio < 0.12:
            # Si hay pocas líneas con puntos suspensivos y muchos datos numéricos,
            # probablemente es una tabla de datos, no un TOC
            data_table_indicators = len(re.findall(r'\d+\.\d+', text))  # Valores decimales
            if data_table_indicators > 3:
                logger.debug("Detected as data table, not TOC")
                return False
        
        # Determine if this is a TOC section
        return (
            toc_indicators >= 2 or 
            (dots_numbers_ratio > 0.15 and total_lines > 5) or
            (page_number_ratio > 0.3 and total_lines > 5) or
            (figure_table_ratio > 0.4 and total_lines > 3) or
            ('table of contents' in text.lower()) or
            ('índice' in text.lower() and dots_numbers_ratio > 0.1)
        )
    
    def find_structure_boundaries(self, text: str, start: int, end: int) -> Dict[str, int]:
        """
        Encuentra los límites de estructuras que deberían preservarse intactas.
        
        Args:
            text: Texto completo
            start: Posición de inicio
            end: Posición de fin
            
        Returns:
            Diccionario con límites de estructuras encontradas
        """
        structures = {}
        
        # Ampliar el rango de búsqueda para capturar estructuras completas
        search_start = max(0, start - 200)
        search_end = min(len(text), end + 200)
        search_text = text[search_start:search_end]
        
        # Buscar estructuras a preservar
        for start_pattern, end_pattern in self.preserve_structures:
            for start_match in start_pattern.finditer(search_text):
                struct_start = search_start + start_match.start()
                
                # Si no hay patrón de fin explícito, usar el próximo salto de párrafo
                if end_pattern is None:
                    end_match = re.search(r'\n\s*\n', search_text[start_match.end():])
                    if end_match:
                        struct_end = search_start + start_match.end() + end_match.end()
                    else:
                        # Si no hay salto de párrafo, extender hasta el final del chunk
                        struct_end = search_end
                else:
                    # Buscar el patrón de fin correspondiente
                    end_match = end_pattern.search(search_text[start_match.end():])
                    if end_match:
                        struct_end = search_start + start_match.end() + end_match.end()
                    else:
                        # Si no se encuentra el fin, asumir que termina en el próximo salto de párrafo
                        paragraph_end = re.search(r'\n\s*\n', search_text[start_match.end():])
                        if paragraph_end:
                            struct_end = search_start + start_match.end() + paragraph_end.end()
                        else:
                            struct_end = search_end
                
                # Solo añadir si la estructura está dentro o cerca de nuestro rango de interés
                if (struct_start <= end and struct_end >= start):
                    structures[f"struct_{len(structures)}"] = {
                        "start": struct_start,
                        "end": struct_end,
                        "type": start_pattern.pattern
                    }
        
        return structures
    
    def create_chunks(self, text: str) -> List[Dict]:
        """
        Divide el texto en chunks respetando el tamaño máximo y el solapamiento.
        Mejorado para respetar divisiones naturales y estructuras como tablas y listas.
        
        Args:
            text: Texto a dividir en chunks
            
        Returns:
            Lista de chunks con su contenido y metadatos
        """
        # Preprocesar texto
        text = self.preprocess_text(text)
        
        chunks = []
        start = 0
        
        # Compilar todos los patrones de divisiones naturales en una sola expresión
        natural_breaks_pattern = '|'.join(p.pattern for p in self.natural_breaks)
        if natural_breaks_pattern:
            natural_breaks_regex = re.compile(natural_breaks_pattern)
        else:
            natural_breaks_regex = None
        
        # Patrones para delimitar chunks de forma más natural
        paragraph_delimiter = re.compile(r'\n\s*\n')
        sentence_delimiter = re.compile(r'(?<=[.!?])\s+')
        
        while start < len(text):
            # Determinar punto final del chunk actual (sin solapamiento)
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                # Buscar estructuras que deberían preservarse intactas
                structures = self.find_structure_boundaries(text, start, end)
                
                # Si hay estructuras que se cortan, ajustar límites
                for struct_info in structures.values():
                    # Si una estructura comienza antes del final pero termina después,
                    # decidir si incluirla completa o terminar antes
                    if struct_info['start'] < end and struct_info['end'] > end:
                        # Si la estructura no es muy grande, incluirla completa
                        if struct_info['end'] - struct_info['start'] < self.chunk_size * 0.5:
                            end = struct_info['end']
                        else:
                            # Si es grande, terminar antes de la estructura
                            end = struct_info['start']
                
                # Buscar divisiones naturales cerca del límite del chunk
                if natural_breaks_regex:
                    # Buscar la división natural más cercana al límite ideal
                    search_range = min(end + self.chunk_overlap, len(text)) - max(end - self.chunk_overlap, start)
                    matches = list(natural_breaks_regex.finditer(text, max(end - self.chunk_overlap, start), min(end + self.chunk_overlap, len(text))))
                    
                    if matches:
                        # Encontrar el match más cercano al límite ideal
                        closest_match = min(matches, key=lambda m: abs(m.start() - end))
                        if abs(closest_match.start() - end) < self.chunk_overlap:
                            end = closest_match.start()
                
                # Si no hay división natural, buscar el fin de párrafo más cercano
                if end == min(start + self.chunk_size, len(text)):
                    last_paragraph = paragraph_delimiter.search(text, start, min(end + self.chunk_overlap//2, len(text)))
                    if last_paragraph and last_paragraph.start() > start and last_paragraph.start() < end + self.chunk_overlap//2:
                        end = last_paragraph.start()
                    else:
                        # Si no hay párrafo, buscar el fin de frase más cercano
                        matches = list(sentence_delimiter.finditer(text, max(end - self.chunk_overlap//2, start), min(end + self.chunk_overlap//2, len(text))))
                        if matches:
                            # Encontrar el match más cercano al límite ideal
                            closest_match = min(matches, key=lambda m: abs(m.start() - end))
                            if abs(closest_match.start() - end) < self.chunk_overlap//2:
                                end = closest_match.start()
                        else:
                            # Si no hay match de frase, buscar el último espacio
                            last_space = text.rfind(' ', end - self.chunk_overlap//2, end)
                            if last_space != -1:
                                end = last_space
            
            # Extraer el texto del chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Check if this looks like a table of contents section
                if self.is_toc_section(chunk_text):
                    logger.info(f"Detected TOC section, skipping chunk: {chunk_text[:100]}...")
                    # Skip this chunk and move to next section
                    start = max(end - self.chunk_overlap, start + 1)
                    continue
                
                # Tokenizar para contar tokens
                token_ids = self.tokenizer.encode(chunk_text)
                
                # Calcular hash para detectar duplicados
                chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
                
                # Verificar si este chunk es muy similar a alguno anterior (usando hash)
                duplicate = False
                for existing_chunk in chunks:
                    if existing_chunk['hash'] == chunk_hash:
                        duplicate = True
                        logger.info(f"Skipping duplicate chunk with hash {chunk_hash}")
                        break
                
                if not duplicate:
                    # Crear info del chunk
                    chunk_info = {
                        'content': chunk_text,
                        'start': start,
                        'end': end,
                        'token_count': len(token_ids),
                        'hash': chunk_hash
                    }
                    chunks.append(chunk_info)
            
            # Avanzar al siguiente chunk considerando solapamiento
            start = max(end - self.chunk_overlap, start + 1)
        
        logger.info(f"Documento dividido en {len(chunks)} chunks")
        return chunks
    
    def process_document(self, doc: Union[Dict, DocumentInput], doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa un documento completo: validación, chunking y opcionalmente almacenamiento.
        
        Args:
            doc: Documento a procesar
            doc_id: ID opcional del documento (si ya está creado)
            
        Returns:
            Diccionario con ID del documento y chunks generados
        """
        # Validar documento
        doc = self.validate_document(doc)
        
        # Generar ID si no se proporciona
        doc_id = doc_id or str(uuid.uuid4())
        
        # Crear chunks
        chunks = self.create_chunks(doc.content)
        
        # Añadir metadatos a los chunks
        for i, chunk in enumerate(chunks):
            chunk['doc_id'] = doc_id
            chunk['chunk_number'] = i
            chunk['metadata'] = {
                **doc.metadata,
                'title': doc.title,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            chunk['model_name'] = doc.model_name
        
        # Almacenar chunks si hay repositorio configurado
        if self.chunk_repository:
            try:
                chunk_ids = self.chunk_repository.store_chunks(doc_id, chunks)
                return {
                    'doc_id': doc_id,
                    'title': doc.title,
                    'chunks': len(chunks),
                    'chunk_ids': chunk_ids
                }
            except Exception as e:
                logger.error(f"Error almacenando chunks: {e}")
                raise
        
        # Si no hay repositorio, devolver sólo la información
        return {
            'doc_id': doc_id,
            'title': doc.title,
            'chunks': chunks
        }
    
    def process_from_file(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Procesa un documento directamente desde archivo.
        
        Args:
            file_path: Ruta al archivo
            metadata: Metadatos opcionales a añadir
            
        Returns:
            Resultado del procesamiento
        """
        reader = DocumentReader()
        doc_data = reader.read_document(file_path)
        
        # Combinar metadatos del documento con los proporcionados
        combined_metadata = {
            **doc_data.get('metadata', {}),
            **(metadata or {})
        }
        
        # Preparar entrada de documento
        doc_input = DocumentInput(
            title=metadata.get('title', os.path.basename(file_path)),
            content=doc_data['content'],
            metadata=combined_metadata,
            model_name=metadata.get('model_name', "miniLM")
        )
        
        # Procesar documento
        return self.process_document(doc_input)