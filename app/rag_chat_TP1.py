#!/usr/bin/env python3
"""
Enhanced RAG CV Assistant - Natural conversational interface with RAG + LLM

This script provides a natural chat experience, integrating the RAG system
with external language models (OpenAI, Claude) to generate fluid responses
about CVs without showing technical information to the user.

Features:
1. Natural conversation, with comprehensive logging to console and file
2. Intelligent use of RAG only when necessary
3. Summarized and well-formatted responses
4. Filtrado por persona mediante consulta SQL directa
5. Memoria de sujeto dentro de la conversación
6. Diagnóstico completo para identificar problemas con documentos y chunks
7. Búsqueda multicontextual para preguntas comparativas
8. Detección mejorada de entidades y cambio de contexto
9. Comprensión semántica unificada de distintas formas de preguntar

Usage:
    python cv_assistant.py [--model openai|claude] [--top-k 5] [--debug] [--diagnose-only]
"""

import asyncio
import argparse
import logging
import os
import sys
import textwrap
import shutil
import json
import time
import re
import numpy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Set
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath('..'))

# -----  Config  -------------------------------------------------
MIN_SIMILARITY_CONCRETA = 0.25   # < 25 % se considera "poca evidencia"
FALLBACK_TOP_K          = 8      # n° de chunks a resumir en modo aproximado
LOGS_TO_CONSOLE         = True   # Mostrar logs en consola (true) además de archivo
# ----------------------------------------------------------------

# Configure logging to both file and console
handlers = [
    logging.FileHandler("cv_assistant.log")
]

# Add console handler if enabled
if LOGS_TO_CONSOLE:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger("cv-assistant")

# Suppress low priority logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)

# Import necessary components
from app.core.config.config import Config
from app.core.pipelines.search_pipeline import SearchPipeline

# Import LLM APIs (if available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# ANSI colors for terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Check if the terminal supports colors
def supports_color():
    """Verifies if the terminal supports ANSI colors."""
    if os.name == 'nt':  # Windows
        return False  # By default, disable on Windows
    
    # Check environment variables
    if 'NO_COLOR' in os.environ:
        return False
    
    if not sys.stdout.isatty():
        return False
    
    return True

# Disable colors if there's no support
if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, "")

# Helper function to log SQL queries
def log_sql(query: str, params: Any = None):
    """
    Log SQL queries with parameters for debugging.
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
    """
    formatted_query = query.replace("\n", " ").strip()
    if params:
        logger.debug(f"SQL: {formatted_query} - Params: {params}")
    else:
        logger.debug(f"SQL: {formatted_query}")

# Define TypedDict for conversation state
class ConversationState(TypedDict):
    last_people: List[str]
    last_topic: str
    recent_queries: List[Dict[str, Any]]
    comparative_mode: bool

class EnhancedCVAssistant:
    """Enhanced conversational interface that integrates RAG with external LLMs."""
    
    def __init__(
        self, 
        model: str = 'openai',
        top_k: int = 5,
        use_localhost: bool = True,
        max_conversation_history: int = 10
    ):
        """
        Initializes the enhanced CV assistant.
        
        Args:
            model: Model to use ('openai' or 'claude')
            top_k: Number of results to return from RAG
            use_localhost: If True, uses localhost instead of postgres_pgvector
            max_conversation_history: Maximum number of messages to keep in history
        """
        self.config = Config()
        
        # Use localhost for the database if specified
        if use_localhost:
            logger.info("Using localhost to connect to PostgreSQL")
            self.config.DB_HOST = "localhost"
        
        # Search configuration
        self.top_k = top_k
        self.model_name = model
        self.max_conversation_history = max_conversation_history
        
        # Initialize components
        self.search_pipeline = None
        
        # Conversation history
        self.conversation = []
        
        # Get terminal size
        self.term_width, self.term_height = shutil.get_terminal_size()
        
        # Validate API availability
        self.openai_available = self._check_openai_availability()
        self.claude_available = self._check_claude_availability()
        
        # Determine the model to use according to availability
        self.active_model = self._determine_active_model()
        
        # Initialize document ID mappings for each person
        self.person_doc_ids = {
            "pedro": [],
            "jorge": [],
            "leonardo": []
        }
        
        # Conversation memory for active people
        self.active_people: List[str] = []
        
        # Conversation state for thread-level subject memory
        self.conversation_state: ConversationState = {
            "last_people": [],
            "last_topic": "",
            "recent_queries": [],
            "comparative_mode": False
        }
        
        # Nombre completo de personas para referencias más naturales
        self.person_full_names = {
            "pedro": "Pedro Pérez",
            "jorge": "Jorge Hernán Cuenca Marín",
            "leonardo": "Leonardo Ortiz Arismendi"
        }
        
        logger.info(f"Enhanced CV Assistant initialized (model: {self.active_model}, top-k: {top_k})")
    
    def _check_openai_availability(self) -> bool:
        """Verifies if OpenAI is available and configured."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not installed")
            return False
        
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return False
        
        # Initialize client
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized correctly")
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return False
    
    def _check_claude_availability(self) -> bool:
        """Verifies if Claude is available and configured."""
        if not CLAUDE_AVAILABLE:
            logger.warning("Anthropic library not installed")
            return False
        
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            return False
        
        # Initialize client
        try:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude client initialized correctly")
            return True
        except Exception as e:
            logger.error(f"Error initializing Claude client: {e}")
            return False
    
    def _determine_active_model(self) -> str:
        """Determines which model to use based on availability."""
        # Check first preference
        if self.model_name == 'openai' and self.openai_available:
            return 'openai'
        elif self.model_name == 'claude' and self.claude_available:
            return 'claude'
        
        # If first preference is not available, try alternative
        if self.openai_available:
            return 'openai'
        elif self.claude_available:
            return 'claude'
        
        # If none are available, use only RAG
        logger.warning("No external LLM available. Using only RAG.")
        return 'rag-only'
    
    async def initialize(self) -> bool:
        """Initializes the RAG system components."""
        logger.info("Initializing components...")
        
        try:
            # Create search pipeline
            self.search_pipeline = SearchPipeline(config=self.config)
            
            # Crear índice para title si no existe
            await self._ensure_title_index()
            
            # Load document ID mappings from the database
            await self._load_document_mappings()
            
            logger.info("Components initialized correctly")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _ensure_title_index(self):
        """
        Crea un índice sobre LOWER(title) si no existe ya.
        """
        try:
            if hasattr(self.search_pipeline, 'embedding_repository') and hasattr(self.search_pipeline.embedding_repository, '_get_pool'):
                pool = await self.search_pipeline.embedding_repository._get_pool()
                async with pool.acquire() as conn:
                    # Crear índice si no existe
                    await conn.execute("""
                    CREATE INDEX IF NOT EXISTS documents_title_idx
                        ON documents (LOWER(title));
                    """)
                    logger.info("Índice sobre LOWER(title) creado o ya existente")
        except Exception as e:
            logger.error(f"Error creando índice: {e}")
    
    async def _load_document_mappings(self):
        """
        Carga los doc_ids para cada persona buscando patrones en los títulos y contenido.
        Implementa búsquedas más flexibles usando patrones LIKE/ILIKE.
        """
        try:
            logger.info("Cargando mapeos de documentos por nombre en títulos...")
            if hasattr(self.search_pipeline, 'embedding_repository') and hasattr(self.search_pipeline.embedding_repository, '_get_pool'):
                pool = await self.search_pipeline.embedding_repository._get_pool()
                async with pool.acquire() as conn:
                    # Reiniciar mapeos
                    self.person_doc_ids = {
                        "pedro": [],
                        "jorge": [],
                        "leonardo": []
                    }
                    
                    # Listar todos los documentos para depuración
                    all_docs_query = "SELECT doc_id, title FROM documents"
                    log_sql(all_docs_query)
                    all_docs = await conn.fetch(all_docs_query)
                    
                    logger.info(f"Total documentos encontrados: {len(all_docs)}")
                    for doc in all_docs:
                        logger.info(f"Documento: {doc['doc_id']} - Título: {doc['title']}")
                    
                    # Buscar utilizando patrones más flexibles para cada persona
                    for person in ["pedro", "jorge", "leonardo"]:
                        # 1. Primero buscar por título exacto (CV-Name.pdf)
                        exact_title = f"CV-{person.capitalize()}.pdf"
                        exact_query = "SELECT doc_id FROM documents WHERE title = $1"
                        log_sql(exact_query, [exact_title])
                        exact_docs = await conn.fetch(exact_query, exact_title)
                        
                        # 2. Si no encuentra, buscar utilizando ILIKE con patrón %name%
                        if not exact_docs:
                            logger.info(f"No se encontró título exacto para {person}, buscando con patrón...")
                            pattern_query = "SELECT doc_id FROM documents WHERE LOWER(title) LIKE $1"
                            pattern = f"%{person}%"
                            log_sql(pattern_query, [pattern])
                            pattern_docs = await conn.fetch(pattern_query, pattern)
                            
                            if pattern_docs:
                                logger.info(f"Encontrados {len(pattern_docs)} documentos con patrón '{pattern}' para {person}")
                                self.person_doc_ids[person] = [row['doc_id'] for row in pattern_docs]
                            else:
                                # 3. Si aún no encuentra, buscar en contenido de chunks
                                logger.warning(f"No se encontraron documentos para {person} por título, buscando en contenido...")
                                content_query = """
                                SELECT DISTINCT d.doc_id
                                FROM chunks c
                                JOIN documents d ON c.doc_id = d.doc_id
                                WHERE LOWER(c.content) LIKE $1
                                LIMIT 5
                                """
                                content_pattern = f"%{person}%"
                                log_sql(content_query, [content_pattern])
                                content_docs = await conn.fetch(content_query, content_pattern)
                                
                                if content_docs:
                                    logger.info(f"Encontrados {len(content_docs)} documentos mencionando '{person}' en contenido")
                                    self.person_doc_ids[person] = [row['doc_id'] for row in content_docs]
                                else:
                                    logger.error(f"⚠️ NO SE ENCONTRÓ NINGUNA REFERENCIA A {person.upper()}")
                        else:
                            logger.info(f"Encontrado documento con título exacto para {person}: {exact_title}")
                            self.person_doc_ids[person] = [row['doc_id'] for row in exact_docs]
                        
                        # Verificar chunks para los doc_ids encontrados
                        if self.person_doc_ids[person]:
                            for doc_id in self.person_doc_ids[person]:
                                chunks_count_query = "SELECT COUNT(*) FROM chunks WHERE doc_id = $1"
                                chunks_count = await conn.fetchval(chunks_count_query, doc_id)
                                logger.info(f"El documento {doc_id} para {person} tiene {chunks_count} chunks")
                                
                                # Si hay chunks, mostrar algunos ejemplos
                                if chunks_count > 0:
                                    sample_query = """
                                    SELECT chunk_id, LEFT(content, 100) as preview 
                                    FROM chunks 
                                    WHERE doc_id = $1 
                                    LIMIT 3
                                    """
                                    log_sql(sample_query, [doc_id])
                                    samples = await conn.fetch(sample_query, doc_id)
                                    for sample in samples:
                                        logger.info(f"  Chunk {sample['chunk_id']} preview: {sample['preview']}...")
                                else:
                                    logger.warning(f"⚠️ El documento {doc_id} para {person} NO TIENE CHUNKS")
                        
                    # Registrar los mapeos para depuración
                    logger.info(f"Mapeos finales de documentos:")
                    for person, doc_ids in self.person_doc_ids.items():
                        if doc_ids:
                            logger.info(f"{person.capitalize()}: {doc_ids}")
                            # Mostrar títulos correspondientes
                            for doc_id in doc_ids:
                                doc_title_query = "SELECT title FROM documents WHERE doc_id = $1"
                                doc_title = await conn.fetchval(doc_title_query, doc_id)
                                logger.info(f"  - {doc_id} maps to title: {doc_title}")
                        else:
                            logger.warning(f"⚠️ No se encontraron documentos para {person.upper()}")
        except Exception as e:
            logger.error(f"Error loading document mappings: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Función de diagnóstico completo
    async def run_diagnostics(self):
        """
        Ejecuta diagnósticos completos sobre la base de datos para identificar
        problemas con documentos, chunks y embeddings.
        """
        try:
            logger.info("======= INICIANDO DIAGNÓSTICO COMPLETO =======")
            
            if not hasattr(self.search_pipeline, 'embedding_repository'):
                logger.error("❌ Embedding repository no disponible")
                return
                
            pool = await self.search_pipeline.embedding_repository._get_pool()
            
            async with pool.acquire() as conn:
                # 1. Verificar tabla documents
                doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
                logger.info(f"Total de documentos: {doc_count}")
                
                # 2. Verificar tabla chunks
                chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
                logger.info(f"Total de chunks: {chunk_count}")
                
                # 3. Verificar tabla embeddings
                emb_count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
                logger.info(f"Total de embeddings: {emb_count}")
                
                # 4. Buscar documentos sin chunks
                orphan_docs_query = """
                SELECT d.doc_id, d.title 
                FROM documents d
                LEFT JOIN chunks c ON d.doc_id = c.doc_id
                GROUP BY d.doc_id, d.title
                HAVING COUNT(c.chunk_id) = 0
                """
                orphan_docs = await conn.fetch(orphan_docs_query)
                if orphan_docs:
                    logger.warning(f"⚠️ Encontrados {len(orphan_docs)} documentos sin chunks:")
                    for doc in orphan_docs:
                        logger.warning(f"  - {doc['doc_id']}: {doc['title']}")
                else:
                    logger.info("✅ Todos los documentos tienen chunks")
                
                # 5. Buscar chunks sin embeddings
                orphan_chunks_query = """
                SELECT c.chunk_id, c.doc_id, LEFT(c.content, 50) as preview
                FROM chunks c
                LEFT JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE e.embedding_id IS NULL
                LIMIT 10
                """
                orphan_chunks = await conn.fetch(orphan_chunks_query)
                if orphan_chunks:
                    logger.warning(f"⚠️ Encontrados chunks sin embeddings (mostrando primeros 10 de {len(orphan_chunks)}):")
                    for chunk in orphan_chunks:
                        logger.warning(f"  - Chunk {chunk['chunk_id']} (doc_id: {chunk['doc_id']}): {chunk['preview']}...")
                else:
                    logger.info("✅ Todos los chunks tienen embeddings")
                
                # 6. Búsqueda específica para Pedro, Jorge y Leonardo
                for name in ["pedro", "jorge", "leonardo"]:
                    # Buscar por título exacto (CV-Name.pdf)
                    exact_title = f"CV-{name.capitalize()}.pdf"
                    exact_doc = await conn.fetchrow(
                        "SELECT doc_id, title FROM documents WHERE title = $1", 
                        exact_title
                    )
                    
                    if exact_doc:
                        logger.info(f"✅ Encontrado documento para {name} con título exacto: {exact_doc['title']} (doc_id: {exact_doc['doc_id']})")
                        
                        # Verificar chunks para este documento
                        doc_chunks = await conn.fetch(
                            "SELECT chunk_id, LEFT(content, 50) as preview FROM chunks WHERE doc_id = $1 LIMIT 3",
                            exact_doc['doc_id']
                        )
                        
                        if doc_chunks:
                            logger.info(f"Muestra de chunks para {name}:")
                            for chunk in doc_chunks:
                                logger.info(f"  - Chunk {chunk['chunk_id']}: {chunk['preview']}...")
                        else:
                            logger.warning(f"⚠️ No se encontraron chunks para el documento de {name}")
                    else:
                        # Buscar documentos que contengan el nombre en el título
                        pattern = f"%{name}%"
                        similar_docs = await conn.fetch(
                            "SELECT doc_id, title FROM documents WHERE LOWER(title) LIKE $1",
                            pattern
                        )
                        
                        if similar_docs:
                            logger.info(f"Documentos con '{name}' en el título:")
                            for doc in similar_docs:
                                logger.info(f"  - {doc['doc_id']}: {doc['title']}")
                        else:
                            logger.warning(f"⚠️ No se encontraron documentos para {name}")
                            
                            # Buscar en contenido de chunks
                            content_docs = await conn.fetch("""
                            SELECT DISTINCT d.doc_id, d.title
                            FROM chunks c
                            JOIN documents d ON c.doc_id = d.doc_id
                            WHERE LOWER(c.content) LIKE $1
                            LIMIT 5
                            """, f"%{name}%")
                            
                            if content_docs:
                                logger.info(f"Documentos con '{name}' mencionado en contenido:")
                                for doc in content_docs:
                                    logger.info(f"  - {doc['doc_id']}: {doc['title']}")
                            else:
                                logger.error(f"❌ NO SE ENCONTRÓ NINGUNA MENCIÓN DE {name.upper()} EN LA BASE DE DATOS")
                
                logger.info("======= DIAGNÓSTICO COMPLETO FINALIZADO =======")
                
        except Exception as e:
            logger.error(f"Error durante diagnóstico: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # ---------------------------------------------------------------------
    # Helper: devuelve todos los doc_id cuyo título contenga una sub-cadena
    # ---------------------------------------------------------------------
    async def _get_doc_ids_by_title(
        self,
        conn,
        pattern: str,            # ej. 'CV-Pedro.pdf'  o  '%pedro%'
        case_insensitive: bool = True
    ) -> List[str]:
        """
        Devuelve una lista de doc_id cuyos títulos coinciden con `pattern`.
        Usa ILIKE por defecto para ignorar mayúsculas/minúsculas.

        Ejemplos:
            await _get_doc_ids_by_title(conn, 'CV-Pedro.pdf')
            await _get_doc_ids_by_title(conn, '%pedro%', True)
        """
        op = "ILIKE" if case_insensitive else "LIKE"
        query = f"SELECT doc_id FROM documents WHERE title {op} $1"
        log_sql(query, [pattern])
        rows = await conn.fetch(query, pattern)
        return [r["doc_id"] for r in rows]
    
    async def close(self):
        """Closes connections and cleans up resources."""
        if self.search_pipeline:
            await self.search_pipeline.close()
        logger.info("CV Assistant resources closed")
    
    def _detect_person_in_query(self, query: str) -> List[str]:
        """
        Detecta qué personas son mencionadas en la consulta con mejor reconocimiento
        de variantes y sinónimos.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Lista de personas mencionadas (pedro, jorge, leonardo)
        """
        people = []
        query_lower = query.lower()
        
        # Mejorado: Patrones más flexibles para detectar nombres con variantes
        patterns = {
            'pedro': ['pedro', 'pérez', 'pedrito'],
            'jorge': ['jorge', 'hernán', 'hernan', 'cuenca', 'marín', 'marin'],
            'leonardo': ['leonardo', 'leo', 'ortiz', 'arismendi']
        }
        
        # MEJORA: Detector basado en patrones más flexible
        for person, keywords in patterns.items():
            for keyword in keywords:
                # Verificar si la palabra está completa (no como parte de otra)
                if re.search(r'\b' + keyword + r'\b', query_lower):
                    if person not in people:
                        people.append(person)
                    break  # Si ya encontramos una coincidencia, no buscamos más para esta persona
        
        # MEJORA: Detección más avanzada para consultas comparativas
        if 'los tres' in query_lower or 'los 3' in query_lower or 'todos' in query_lower:
            # Este es un caso especial de comparación
            self.conversation_state["comparative_mode"] = True
            return ["pedro", "jorge", "leonardo"]
        
        # Casos especiales para preguntas implícitas
        if 'comparar' in query_lower or 'compara' in query_lower or 'comparación' in query_lower:
            self.conversation_state["comparative_mode"] = True
            if not people:  # Si no se detectaron personas específicas
                return ["pedro", "jorge", "leonardo"]  # Asumir que quiere comparar a todos
        
        # MEJORA: Detección de "él" o "ella" para referencias anafóricas
        if ('él' in query_lower or 'ella' in query_lower or 'este' in query_lower or 
            'esta' in query_lower or 'su' in query_lower) and self.active_people:
            # Mantener el último contexto si hay una referencia anafórica
            logger.info(f"Referencia anafórica detectada, manteniendo personas activas: {self.active_people}")
            return self.active_people
            
        return people
    
    # Method to update active people
    def _update_active_people(self, mentioned: List[str]):
        """
        Actualiza la lista de personas activas en la conversación con manejo
        más inteligente de cambios de contexto.
        
        Args:
            mentioned: Lista de personas mencionadas en la consulta actual
        """
        if mentioned:
            # Si estamos en modo comparativo y se mencionan personas específicas, 
            # actualizar el modo comparativo
            if self.conversation_state["comparative_mode"] and len(mentioned) == 1:
                # Salir del modo comparativo si solo se pregunta por una persona
                self.conversation_state["comparative_mode"] = False
            
            # Si hay menciones explícitas, reemplaza las personas activas
            self.active_people = mentioned
            # Actualiza también el estado de la conversación
            self.conversation_state["last_people"] = mentioned
            logger.info(f"Actualizando personas activas a: {mentioned}")
        elif not self.active_people and len(self.conversation) > 0:
            # Si no hay personas activas pero hay historial, intentar inferir del contexto
            recent_people = self.conversation_state.get("last_people", [])
            if recent_people:
                logger.info(f"Inferiendo personas del contexto histórico: {recent_people}")
                self.active_people = recent_people
    
    # Detección de tópico para mejorar contexto
    def _detect_topic_in_query(self, query: str) -> str:
        """
        Detecta el tópico principal de la consulta con mejor reconocimiento de 
        variantes y sinónimos.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Tópico detectado o cadena vacía
        """
        query_lower = query.lower()
        
        # MEJORA: Temas más detallados con más sinónimos para mejor cobertura
        topics = {
            "experiencia": [
                "experiencia", "trabajo", "profesional", "empleo", "trayectoria", 
                "laboral", "empresa", "cargo", "puesto", "rol", "función", 
                "responsabilidades", "labor", "ocupación", "desempeño"
            ],
            "educación": [
                "educación", "estudios", "formación", "universidad", "título", 
                "carrera", "académica", "diploma", "grado", "certificado", 
                "licenciatura", "maestría", "doctorado", "postgrado", "curso"
            ],
            "habilidades": [
                "habilidades", "skills", "competencias", "conocimientos", 
                "aptitudes", "tecnologías", "herramientas", "lenguajes", 
                "capacidades", "destrezas", "técnicas", "metodologías",
                "frameworks", "plataformas", "especialidades", "expertise"
            ],
            "proyectos": [
                "proyectos", "logros", "éxitos", "achievements", "resultados",
                "casos", "implementaciones", "desarrollo", "soluciones", 
                "aplicaciones", "sistemas", "iniciativas", "contribuciones"
            ],
            "contacto": [
                "contacto", "email", "teléfono", "dirección", "datos", 
                "información personal", "referencias", "contactar", "ubicación"
            ]
        }
        
        # MEJORA: Contar ocurrencias de palabras clave por tema
        topic_counts = {topic: 0 for topic in topics}
        
        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query_lower:
                    topic_counts[topic] += 1
        
        # Elegir el tema con más coincidencias
        max_count = max(topic_counts.values())
        if max_count > 0:
            # Si hay empate, elegir el primer tema con ese conteo
            for topic, count in topic_counts.items():
                if count == max_count:
                    return topic
        
        return ""
    
    def _get_person_doc_ids_from_query(self, query: str) -> List[str]:
        """
        Detecta la persona mencionada en la consulta y devuelve sus doc_ids.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Lista de doc_ids de la persona mencionada
        """
        # Detectar persona en la consulta (busca menciones de pedro, jorge o leonardo)
        mentioned_people = self._detect_person_in_query(query)
        
        # MEJORA: Verificar si es una pregunta comparativa
        is_comparative = (
            self.conversation_state.get("comparative_mode", False) or
            'comparar' in query.lower() or 'compara' in query.lower() or
            'los tres' in query.lower() or 'todos los' in query.lower()
        )
        
        # Si es pregunta comparativa, usar todos los perfiles
        if is_comparative and not mentioned_people:
            mentioned_people = ["pedro", "jorge", "leonardo"]
            logger.info("Detectada consulta comparativa, usando todos los perfiles")
            self.conversation_state["comparative_mode"] = True
            
        # Si no hay personas mencionadas, usar personas activas de la conversación
        target_people = mentioned_people or self.active_people
        
        # Si hay personas objetivo, actualizar el estado de la conversación
        if mentioned_people:
            self._update_active_people(mentioned_people)
        
        # Obtener doc_ids para las personas objetivo
        doc_ids = []
        for person in target_people:
            person_docs = self.person_doc_ids.get(person, [])
            if person_docs:
                doc_ids.extend(person_docs)
                logger.info(f"Using document_ids for {person}: {person_docs}")
            else:
                logger.warning(f"No document IDs found for {person}")
        
        return doc_ids
    
    async def _perform_expanded_search(self, query: str, doc_ids: List[str], topic: str = "") -> Dict[str, Any]:
        """
        Realiza una búsqueda expandida para obtener información más relevante cuando
        la búsqueda original solo devuelve información básica.
        
        Args:
            query: Consulta original
            doc_ids: Lista de IDs de documentos de la persona
            topic: Tópico detectado para enfocar mejor la búsqueda
            
        Returns:
            Resultados expandidos de la búsqueda
        """
        try:
            # Determinar consultas específicas según el tópico o usar consultas genéricas
            expanded_queries = []
            
            if topic == "experiencia":
                expanded_queries = [
                    "experiencia laboral",
                    "trabajo profesional",
                    "historial empresas",
                    "cargo posición",
                    "trayectoria profesional",
                    "logros laborales"
                ]
            elif topic == "educación":
                expanded_queries = [
                    "formación académica",
                    "universidad estudios",
                    "títulos educación",
                    "diplomas certificaciones",
                    "cursos especializados"
                ]
            elif topic == "habilidades":
                expanded_queries = [
                    "habilidades competencias",
                    "conocimientos técnicos",
                    "herramientas tecnologías", 
                    "lenguajes programación",
                    "capacidades técnicas"
                ]
            else:
                # Consultas genéricas para obtener información variada
                expanded_queries = [
                    "experiencia profesional",
                    "formación académica",
                    "habilidades competencias",
                    "información relevante",
                    "perfil profesional",
                    "logros destacados"
                ]
            
            # Acceder a la conexión de base de datos
            pool = await self.search_pipeline.embedding_repository._get_pool()
            
            # Resultados combinados
            all_chunks = []
            
            # Realizar búsquedas por contenido para cada consulta expandida
            async with pool.acquire() as conn:
                for expanded_query in expanded_queries:
                    # Búsqueda directa por contenido
                    content_query = """
                    SELECT c.chunk_id, c.content, c.doc_id, d.title
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.doc_id = ANY($1)
                    AND (
                        c.content ILIKE $2
                        OR c.content ILIKE $3
                    )
                    ORDER BY LENGTH(c.content) DESC
                    LIMIT 3
                    """
                    
                    # Dividir la consulta expandida y usar los términos para la búsqueda
                    terms = expanded_query.split()
                    if len(terms) >= 2:
                        pattern1 = f"%{terms[0]}%"
                        pattern2 = f"%{terms[1]}%" if len(terms) > 1 else pattern1
                        
                        rows = await conn.fetch(content_query, doc_ids, pattern1, pattern2)
                        
                        # Procesar y agregar resultados
                        for row in rows:
                            # Verificar que no sea solo información básica
                            if len(row['content']) > 100:  # Solo chunks sustanciales
                                chunk = {
                                    'chunk_id': row['chunk_id'],
                                    'content': row['content'],
                                    'similarity': 0.7,  # Prioridad alta al ser búsqueda dirigida
                                    'title': row['title'],
                                    'doc_id': row['doc_id'],
                                    'metadata': {}
                                }
                                # Evitar duplicados
                                if not any(c['chunk_id'] == chunk['chunk_id'] for c in all_chunks):
                                    all_chunks.append(chunk)
            
            # Si encontramos chunks relevantes, crear contexto y devolver resultados
            if all_chunks:
                logger.info(f"Búsqueda expandida encontró {len(all_chunks)} chunks más informativos")
                
                # Construcción de contexto
                context = ""
                for chunk in all_chunks:
                    doc_header = f"[Documento: {chunk['title']}]\n"
                    context += doc_header + chunk['content'] + "\n\n"
                
                return {
                    'chunks': all_chunks,
                    'context': context,
                    'max_similarity': 0.7,  # Valor razonable para estos resultados
                    'metadata': {
                        'filtered_by': 'expanded_search',
                        'doc_ids': doc_ids,
                        'results_count': len(all_chunks)
                    }
                }
            
            # Si no hay resultados, devolver vacío
            return {}
            
        except Exception as e:
            logger.error(f"Error en búsqueda expandida: {e}")
            return {}
    
    async def _perform_filtered_search(self, query: str, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Realiza una búsqueda vectorial filtrada por doc_ids específicos.
        Versión corregida para funcionar correctamente con pgvector.
        
        Args:
            query: Consulta del usuario
            doc_ids: Lista de document_ids para filtrar
            
        Returns:
            Resultados de la búsqueda
        """
        try:
            logger.info(f"Ejecutando búsqueda filtrada para query: '{query}'")
            
            if not self.search_pipeline.embedding_repository:
                logger.error("❌ Embedding repository no disponible")
                return {}
            
            # Verificar que doc_ids no esté vacío
            if not doc_ids:
                logger.warning("❌ Se proporcionaron doc_ids vacíos, devolviendo resultados vacíos")
                return {}
            
            # Acceder a la conexión de base de datos
            pool = await self.search_pipeline.embedding_repository._get_pool()
            
            async with pool.acquire() as conn:
                # Registrar los doc_ids que se están usando
                logger.info(f"Filtering search by doc_ids: {doc_ids}")
                
                # Verificar que haya chunks para los doc_ids proporcionados
                chunks_check_query = """
                SELECT COUNT(*) FROM chunks WHERE doc_id = ANY($1)
                """
                chunks_count = await conn.fetchval(chunks_check_query, doc_ids)
                logger.info(f"Encontrados {chunks_count} chunks para los doc_ids proporcionados")
                
                if chunks_count == 0:
                    logger.warning(f"⚠️ No hay chunks para los doc_ids: {doc_ids}")
                    return {}
                
                # Calcular el embedding de la consulta
                logger.info("Generando embedding para la consulta...")
                embedding = None
                if hasattr(self.search_pipeline, 'embedding_service'):
                    # Corregir verificación de embeddings
                    embeddings = self.search_pipeline.embedding_service.generate_embeddings([query])
                    if embeddings is not None and len(embeddings) > 0:
                        embedding = embeddings[0]
                        logger.info(f"Embedding generado: {len(embedding)} dimensiones")
                    else:
                        logger.error("❌ No se pudo generar embedding o está vacío")
                
                if embedding is not None:
                    # MEJORA: Optimización para consultas más relevantes
                    # Identificar si es una consulta comparativa
                    is_comparative = self.conversation_state.get("comparative_mode", False)
                    
                    # MEJORA: Usar un top_k dinámico en consultas comparativas para asegurar
                    # que tenemos suficientes resultados de cada documento
                    effective_top_k = self.top_k * 2 if is_comparative else self.top_k
                    
                    # CORRECCIÓN: Usar el formato correcto para pgvector
                    # Consulta SQL modificada que usa la sintaxis de pgvector para vector
                    query_sql = """
                    SELECT 
                        e.embedding_id, e.chunk_id, e.model_name,
                        1 - (e.embedding <=> $1::vector) AS similarity,
                        c.content, c.metadata, d.title, d.doc_id
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.chunk_id
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE d.doc_id = ANY($2)
                    ORDER BY similarity DESC
                    LIMIT $3
                    """
                    
                    # CORRECCIÓN: Convertir el embedding a formato JSON para pgvector
                    embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
                    
                    logger.info(f"Ejecutando búsqueda vectorial para doc_ids: {doc_ids}")
                    
                    # CORRECCIÓN: Usar el embedding como JSON string
                    rows = await conn.fetch(query_sql, embedding_json, doc_ids, effective_top_k)
                    
                    # Verificar si se obtuvieron resultados
                    if not rows:
                        logger.warning(f"⚠️ No se encontraron resultados para los doc_ids: {doc_ids}")
                        
                        # Consulta diagnóstica para ver qué está disponible
                        sample_query = """
                        SELECT c.chunk_id, d.doc_id, d.title, LEFT(c.content, 100) as preview
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        WHERE d.doc_id = ANY($1)
                        LIMIT 5
                        """
                        samples = await conn.fetch(sample_query, doc_ids)
                        logger.info(f"Muestras de chunks disponibles para estos doc_ids:")
                        for sample in samples:
                            logger.info(f"  - Chunk {sample['chunk_id']}: '{sample['preview']}...' (doc_id: {sample['doc_id']})")
                        
                        return {}
                    
                    # Procesar resultados
                    logger.info(f"✅ Se encontraron {len(rows)} resultados")
                    
                    # Calcular el valor máximo de similitud
                    max_sim = max([r['similarity'] for r in rows], default=0.0)
                    logger.info(f"Similitud máxima: {max_sim:.4f}")
                    
                    # Procesar los resultados
                    chunks = []
                    for row in rows:
                        chunk = {
                            'embedding_id': row['embedding_id'],
                            'chunk_id': row['chunk_id'],
                            'model_name': row['model_name'],
                            'similarity': row['similarity'],
                            'content': row['content'],
                            'metadata': row['metadata'] if isinstance(row['metadata'], dict) else 
                                    json.loads(row['metadata']) if row['metadata'] else {},
                            'title': row['title'],
                            'doc_id': row['doc_id']
                        }
                        chunks.append(chunk)
                    
                    # MEJORA: Para consultas comparativas, asegurar representación equitativa
                    if is_comparative and len(doc_ids) > 1:
                        # Obtener representación equitativa de cada persona
                        balanced_chunks = self._balance_comparative_results(chunks, doc_ids)
                        if balanced_chunks:
                            chunks = balanced_chunks
                            logger.info(f"Reordenados chunks para consulta comparativa: {len(chunks)} chunks balanceados")
                    
                    # Registrar los resultados para depuración
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Resultado {i+1}: doc_id={chunk['doc_id']}, título={chunk['title']}, similitud={chunk['similarity']:.4f}")
                        content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                        logger.info(f"Vista previa: {content_preview}")
                    
                    # Construir contexto a partir de los chunks
                    context = ""
                    for chunk in chunks:
                        if 'content' in chunk and chunk['content']:
                            # Agregar un identificador del documento fuente
                            doc_header = f"[Documento: {chunk['title']}]\n"
                            context += doc_header + chunk['content'] + "\n\n"
                    
                    return {
                        'chunks': chunks,
                        'context': context,
                        'max_similarity': max_sim,
                        'metadata': {
                            'filtered_by': 'doc_id',
                            'doc_ids': doc_ids,
                            'results_count': len(chunks),
                            'is_comparative': is_comparative
                        }
                    }
                else:
                    logger.error("❌ No se pudo generar embedding para la consulta")
                    return {}
        except Exception as e:
            logger.error(f"Error en búsqueda filtrada: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # MEJORA: Intentar búsqueda de respaldo simple en caso de error
            try:
                logger.info("Intentando búsqueda simple por contenido textual como respaldo...")
                if pool and doc_ids:
                    async with pool.acquire() as conn:
                        # Búsqueda por palabras clave en lugar de vectorial
                        fallback_query = """
                        SELECT c.chunk_id, c.content, c.doc_id, d.title
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        WHERE c.doc_id = ANY($1)
                        AND (
                            c.content ILIKE $2 
                            OR c.content ILIKE $3
                        )
                        LIMIT $4
                        """
                        
                        # Palabras clave de la consulta
                        keywords = query.split()
                        if len(keywords) > 0:
                            keyword1 = f"%{keywords[0]}%"
                            keyword2 = f"%{keywords[-1]}%" if len(keywords) > 1 else keyword1
                            
                            fallback_rows = await conn.fetch(fallback_query, doc_ids, keyword1, keyword2, self.top_k)
                            
                            if fallback_rows:
                                logger.info(f"Búsqueda de respaldo encontró {len(fallback_rows)} resultados")
                                
                                # Crear respuesta similar al método principal
                                chunks = []
                                for row in fallback_rows:
                                    chunk = {
                                        'chunk_id': row['chunk_id'],
                                        'content': row['content'],
                                        'similarity': 0.5,  # Valor arbitrario
                                        'title': row['title'],
                                        'doc_id': row['doc_id'],
                                        'metadata': {}
                                    }
                                    chunks.append(chunk)
                                
                                # Construir contexto
                                context = ""
                                for chunk in chunks:
                                    doc_header = f"[Documento: {chunk['title']}]\n"
                                    context += doc_header + chunk['content'] + "\n\n"
                                
                                return {
                                    'chunks': chunks,
                                    'context': context,
                                    'max_similarity': 0.5,
                                    'metadata': {
                                        'filtered_by': 'text_fallback',
                                        'doc_ids': doc_ids,
                                        'results_count': len(chunks)
                                    }
                                }
            except Exception as backup_error:
                logger.error(f"Error en búsqueda de respaldo: {backup_error}")
            
            return {} 
    
    def _balance_comparative_results(self, chunks: List[Dict[str, Any]], doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Rebalancea los resultados para asegurar una representación equitativa de cada documento
        en consultas comparativas.
        
        Args:
            chunks: Lista de chunks encontrados
            doc_ids: Lista de document_ids usados en la búsqueda
            
        Returns:
            Lista rebalanceada de chunks
        """
        if not chunks or len(doc_ids) <= 1:
            return chunks
        
        # Agrupar chunks por doc_id
        chunks_by_doc = {}
        for doc_id in doc_ids:
            chunks_by_doc[doc_id] = []
        
        # Clasificar cada chunk en su documento correspondiente
        for chunk in chunks:
            doc_id = chunk.get('doc_id')
            if doc_id in chunks_by_doc:
                chunks_by_doc[doc_id].append(chunk)
            
        # Encontrar el número mínimo de chunks por documento (al menos 1)
        min_chunks_per_doc = max(1, min(len(doc_chunks) for doc_chunks in chunks_by_doc.values() if doc_chunks))
        
        # Para documentos sin chunks, registrarlo
        empty_docs = [doc_id for doc_id, doc_chunks in chunks_by_doc.items() if not doc_chunks]
        if empty_docs:
            logger.warning(f"Los siguientes documentos no tienen chunks en los resultados: {empty_docs}")
        
        # Crear una lista balanceada tomando chunks de cada documento en orden
        balanced = []
        remaining = []
        
        # Primero, tomar igual número de chunks de cada documento
        for doc_id, doc_chunks in chunks_by_doc.items():
            if doc_chunks:
                # Ordenar por similitud y tomar los mejores
                sorted_chunks = sorted(doc_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
                balanced.extend(sorted_chunks[:min_chunks_per_doc])
                remaining.extend(sorted_chunks[min_chunks_per_doc:])
        
        # Luego, agregar los chunks restantes ordenados por similitud
        if remaining:
            sorted_remaining = sorted(remaining, key=lambda x: x.get('similarity', 0), reverse=True)
            balanced.extend(sorted_remaining)
        
        return balanced
    
    async def _perform_rag_search(self, query: str) -> Dict[str, Any]:
        """
        Ejecuta el proceso de búsqueda RAG completo, siguiendo la secuencia:
        1. Identifica la persona mencionada
        2. Obtiene su doc_id de la tabla documents
        3. Filtra chunks por ese doc_id
        4. Realiza la búsqueda vectorial sobre esos chunks
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Resultados de la búsqueda
        """
        if not self.search_pipeline:
            logger.error("Search pipeline not initialized")
            return {}
        
        logger.info(f"Searching in RAG: '{query}'")
        
        try:
            # 1. Detectar personas y actualizar estado de la conversación
            mentioned_people = self._detect_person_in_query(query)
            detected_topic = self._detect_topic_in_query(query)
            
            # MEJORA: Detectar si es una consulta comparativa
            is_comparative = (
                'comparar' in query.lower() or 
                'compara' in query.lower() or 
                'los tres' in query.lower() or 
                'todos' in query.lower() or
                len(mentioned_people) > 1
            )
            
            # Actualizar el estado de conversación
            if is_comparative:
                self.conversation_state["comparative_mode"] = True
                logger.info("Modo comparativo activado por consulta explícita")
            
            # Actualizar personas activas y tema de la conversación
            self._update_active_people(mentioned_people)
            if detected_topic:
                self.conversation_state["last_topic"] = detected_topic
            
            # Guardar la consulta en el historial reciente
            recent_query = {
                "query": query,
                "people": mentioned_people or self.active_people,
                "topic": detected_topic,
                "is_comparative": is_comparative
            }
            self.conversation_state["recent_queries"].append(recent_query)
            # Mantener solo las últimas 5 consultas
            if len(self.conversation_state["recent_queries"]) > 5:
                self.conversation_state["recent_queries"] = self.conversation_state["recent_queries"][-5:]
            
            # Loguear para depuración
            target_people = mentioned_people or self.active_people
            if target_people:
                people_str = ", ".join(target_people)
                logger.info(f"Target people: {people_str}")
            
            # 2. Obtener doc_ids de las personas objetivo
            doc_ids = self._get_person_doc_ids_from_query(query)
            
            # 3. Si hay doc_ids, realizar búsqueda filtrada
            if doc_ids:
                # Realizar búsqueda filtrada por doc_id
                logger.info(f"Realizando búsqueda filtrada con doc_ids: {doc_ids}")
                results = await self._perform_filtered_search(query, doc_ids)
                
                # Si se obtuvieron resultados, verificar si son de calidad
                if results and results.get('chunks'):
                    logger.info(f"Found {len(results.get('chunks', []))} filtered results")
                    
                    # MEJORA: Verificar si los resultados solo contienen información básica
                    # (típicamente ocurre con Leonardo donde solo recupera nombre y contacto)
                    basic_info_patterns = [
                        "REFERENCIAS PERSONALES", 
                        "CEL", 
                        "C.C", 
                        "TELÉFONO", 
                        "CONTACTO",
                        "CELULAR",
                        "EMAIL",
                        "CORREO"
                    ]
                    
                    # Verificar si todos los chunks son información básica de contacto
                    all_basic_info = True
                    for chunk in results.get('chunks', []):
                        content = chunk.get('content', '')
                        # Si al menos un chunk contiene información sustancial, conservamos los resultados
                        if not any(pattern in content for pattern in basic_info_patterns) or len(content) > 150:
                            all_basic_info = False
                            break
                    
                    # Si todos los chunks son información básica, ampliar la búsqueda
                    if all_basic_info and target_people:
                        logger.info(f"Solo se encontró información básica para {target_people}, ampliando búsqueda...")
                        
                        # Búsqueda expandida para obtener más información
                        expanded_results = await self._perform_expanded_search(query, doc_ids, topic=detected_topic)
                        if expanded_results and expanded_results.get('chunks'):
                            # MEJORA: Mezclar algunos resultados básicos con los expandidos para contexto completo
                            combined_chunks = expanded_results.get('chunks', [])
                            # Añadir 1-2 chunks básicos para mantener información de contacto si es necesaria
                            for basic_chunk in results.get('chunks', [])[:2]:  # Solo tomar los 2 mejores chunks básicos
                                if not any(c['chunk_id'] == basic_chunk['chunk_id'] for c in combined_chunks):
                                    combined_chunks.append(basic_chunk)
                            
                            # Reconstruir contexto con chunks combinados
                            context = ""
                            for chunk in combined_chunks:
                                doc_header = f"[Documento: {chunk['title']}]\n"
                                context += doc_header + chunk['content'] + "\n\n"
                            
                            expanded_results['chunks'] = combined_chunks
                            expanded_results['context'] = context
                            return expanded_results
                    
                    # MEJORA: Detectar si hay que hacer una consulta comparativa
                    if is_comparative and len(doc_ids) > 1:
                        # Ya incluido en perform_filtered_search
                        results['metadata']['is_comparative'] = True
                    
                    return results
                else:
                    logger.warning("No results from filtered search, falling back to standard search")
            else:
                logger.warning("No doc_ids found for the mentioned people")
            
            # 4. Si no hay doc_ids o no se obtuvieron resultados, usar búsqueda estándar
            logger.info("Fallback to standard search")
            results = await self.search_pipeline.search(
                query_text=query,
                mode='hybrid',
                top_k=self.top_k,
                strategy='relevance'
            )
            
            # Calcular similitud máxima
            if results and results.get('chunks'):
                max_sim = max([c.get('similarity', 0) for c in results.get('chunks', [])], default=0.0)
                results['max_similarity'] = max_sim
                logger.info(f"Búsqueda estándar encontró {len(results.get('chunks', []))} resultados, max_sim={max_sim:.4f}")
            else:
                logger.warning("La búsqueda estándar no encontró resultados")
            
            return results
                
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    # Helper para resumir chunks
    def _summarize_chunks(self, chunks: List[Dict[str, Any]], max_chars: int = 600) -> str:
        """
        Hace un resumen naïf tomando las primeras líneas de cada chunk.
        (Puedes reemplazar esto por tu propio modelo extractor si quieres).
        """
        summary = []
        for ch in chunks[:FALLBACK_TOP_K]:
            # toma sólo la primera línea de cada chunk
            first_line = ch['content'].splitlines()[0].strip()
            if first_line:
                summary.append(f"• {first_line}")
            if sum(len(s) for s in summary) > max_chars:
                break
        return "\n".join(summary)
    
    def _create_prompt_with_rag_context(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un prompt para el LLM con contexto de RAG e instrucciones estrictas de separación.
        Versión mejorada para manejo de consultas comparativas y transiciones de contexto más naturales.
        
        Args:
            query: Consulta del usuario
            rag_results: Resultados de RAG
            
        Returns:
            Prompt formateado para el LLM
        """
        # Obtener contexto
        context = rag_results.get('context', "")
        
        # Verificar si estamos en modo aproximado
        if rag_results.get('approx_mode', False):
            context = (
                "⚠️ El sistema no encontró datos exactos. A continuación se muestran "
                "fragmentos aproximados; responde **sin inventar nada** y deja claro "
                "que los datos son aproximados.\n\n" + context
            )
        
        # Identificar las personas objetivo
        target_people = self._detect_person_in_query(query) or self.active_people
        
        # MEJORA: Usar nombres completos para una experiencia más natural
        people_str = ""
        if target_people:
            full_names = [self.person_full_names.get(p, p.capitalize()) for p in target_people]
            if len(full_names) == 1:
                people_str = full_names[0]
            elif len(full_names) == 2:
                people_str = f"{full_names[0]} y {full_names[1]}"
            else:
                people_str = ", ".join(full_names[:-1]) + f" y {full_names[-1]}"
        else:
            people_str = "Pedro Pérez, Jorge Hernán Cuenca Marín y Leonardo Ortiz Arismendi"
        
        # Verificar si estamos en modo comparativo
        is_comparative = (
            rag_results.get('metadata', {}).get('is_comparative', False) or 
            self.conversation_state.get("comparative_mode", False)
        )
        
        # Formatear el prompt según el modelo activo
        if self.active_model == 'openai':
            messages = []
            
            # Instrucciones con énfasis en la separación estricta de información
            system_msg = f"""
            Eres un asistente especializado en análisis de hojas de vida, experto y natural en tus respuestas.
            
            INSTRUCCIÓN CRÍTICA DE SEPARACIÓN DE INFORMACIÓN:
            - Si te preguntan sobre Pedro, usa SOLO información de documentos de Pedro (CV-Pedro.pdf).
            - Si te preguntan sobre Jorge, usa SOLO información de documentos de Jorge (CV-Jorge.pdf).
            - Si te preguntan sobre Leonardo, usa SOLO información de documentos de Leonardo (CV-Leonardo.pdf).
            - NUNCA mezcles información entre personas diferentes EXCEPTO en preguntas comparativas.
            
            Cada fragmento de información está marcado con [Documento: nombre-del-documento].
            - Si el documento se llama "CV-Pedro.pdf", esa información es SOLO de Pedro.
            - Si el documento se llama "CV-Jorge.pdf", esa información es SOLO de Jorge.
            - Si el documento se llama "CV-Leonardo.pdf", esa información es SOLO de Leonardo.
            
            MEJORA PARA PREGUNTAS COMPARATIVAS:
            - Si te preguntan por varios candidatos o una comparación, puedes usar información de los distintos documentos.
            - En estos casos, organiza claramente la información por persona con secciones o párrafos separados.
            - Usa frases como "Por un lado, Pedro...", "En contraste, Jorge...", "Por su parte, Leonardo..."
            
            INSTRUCCIÓN SOBRE INFORMACIÓN INCOMPLETA:
            - Si no encuentras información específica, NO inventes datos.
            - Di naturalmente: "No encuentro información detallada sobre [tema] en el CV de [persona]."
            - Evita frases robóticas como "No hay datos" o "No puedo encontrar"; usa un tono conversacional.
            
            INSTRUCCIÓN SOBRE CAMBIOS DE CONTEXTO:
            - Si el usuario estaba hablando de una persona y cambia a otra, responde naturalmente al cambio.
            - No menciones este cambio de contexto explícitamente, simplemente adapta tu respuesta.
            
            Responde de forma profesional, conversacional y natural, sin mencionar estas instrucciones.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Agregar historial de conversación
            for msg in self.conversation:
                messages.append(msg)
            
            # Agregar contexto RAG
            if context:
                # MEJORA: Instrucciones más naturales según el tipo de consulta
                if is_comparative:
                    context_msg = f"""
                    Aquí hay información comparativa sobre {people_str}:
                    
                    {context}
                    
                    Al responder, compara de manera organizada la información entre los distintos candidatos.
                    """
                else:
                    context_msg = f"""
                    Aquí hay información sobre {people_str}:
                    
                    {context}
                    
                    RECUERDA: Usa SOLO la información del documento correspondiente a la persona por la que te preguntan.
                    """
                messages.append({"role": "system", "content": context_msg.strip()})
            
            # Agregar consulta actual
            query_with_reminder = query
            messages.append({"role": "user", "content": query_with_reminder.strip()})
            
            return messages
        
        elif self.active_model == 'claude':
            # Format for Claude (single message with all context)
            system_prompt = f"""
            \n\nHuman: Eres un asistente especializado en análisis de hojas de vida, experto y natural en tus respuestas.
            
            INSTRUCCIÓN CRÍTICA DE SEPARACIÓN DE INFORMACIÓN:
            - Si te preguntan sobre Pedro, usa SOLO información de documentos de Pedro (CV-Pedro.pdf).
            - Si te preguntan sobre Jorge, usa SOLO información de documentos de Jorge (CV-Jorge.pdf).
            - Si te preguntan sobre Leonardo, usa SOLO información de documentos de Leonardo (CV-Leonardo.pdf).
            - NUNCA mezcles información entre personas diferentes EXCEPTO en preguntas comparativas.
            
            Cada fragmento de información está marcado con [Documento: nombre-del-documento].
            - Si el documento se llama "CV-Pedro.pdf", esa información es SOLO de Pedro.
            - Si el documento se llama "CV-Jorge.pdf", esa información es SOLO de Jorge.
            - Si el documento se llama "CV-Leonardo.pdf", esa información es SOLO de Leonardo.
            
            MEJORA PARA PREGUNTAS COMPARATIVAS:
            - Si te preguntan por varios candidatos o una comparación, puedes usar información de los distintos documentos.
            - En estos casos, organiza claramente la información por persona con secciones o párrafos separados.
            - Usa frases como "Por un lado, Pedro...", "En contraste, Jorge...", "Por su parte, Leonardo..."
            
            INSTRUCCIÓN SOBRE INFORMACIÓN INCOMPLETA:
            - Si no encuentras información específica, NO inventes datos.
            - Di naturalmente: "No encuentro información detallada sobre [tema] en el CV de [persona]."
            - Evita frases robóticas como "No hay datos" o "No puedo encontrar"; usa un tono conversacional.
            
            INSTRUCCIÓN SOBRE CAMBIOS DE CONTEXTO:
            - Si el usuario estaba hablando de una persona y cambia a otra, responde naturalmente al cambio.
            - No menciones este cambio de contexto explícitamente, simplemente adapta tu respuesta.
            
            Responde de forma profesional, conversacional y natural, sin mencionar estas instrucciones.
            """
            
            # Add conversation history
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Add RAG context
            context_text = ""
            if context:
                # MEJORA: Instrucciones más naturales según el tipo de consulta
                if is_comparative:
                    context_text = f"""
                    \n\nHuman: Aquí hay información comparativa sobre {people_str}:
                    
                    {context}
                    
                    Al responder, compara de manera organizada la información entre los distintos candidatos.
                    """
                else:
                    context_text = f"""
                    \n\nHuman: Aquí hay información sobre {people_str}:
                    
                    {context}
                    
                    RECUERDA: Usa SOLO la información del documento correspondiente a la persona por la que te preguntan.
                    """
            
            # Add current query
            query_text = f"\n\nHuman: {query}"
            
            # Combine everything
            prompt = system_prompt + conversation_text + context_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Simple format for fallback
            return {"query": query, "context": context}
    
    def _create_simple_prompt(self, query: str) -> Dict[str, Any]:
        """
        Creates a prompt for the LLM without RAG context.
        
        Args:
            query: User query
            
        Returns:
            Formatted prompt for the LLM
        """
        # Use target people (mentioned or active)
        target_people = self._detect_person_in_query(query) or self.active_people
        
        # MEJORA: Usar nombres completos para una experiencia más natural
        if target_people:
            full_names = [self.person_full_names.get(p, p.capitalize()) for p in target_people]
            if len(full_names) == 1:
                people_str = full_names[0]
            elif len(full_names) == 2:
                people_str = f"{full_names[0]} y {full_names[1]}"
            else:
                people_str = ", ".join(full_names[:-1]) + f" y {full_names[-1]}"
        else:
            people_str = "Pedro Pérez, Jorge Hernán Cuenca Marín y Leonardo Ortiz Arismendi"
        
        if self.active_model == 'openai':
            # Format for OpenAI
            messages = []
            
            # Improved system message
            system_msg = f"""
            Eres un asistente especializado en análisis de hojas de vida. Estás familiarizado con los perfiles de {people_str}.
            
            Al responder consultas sobre estos candidatos:
            1. Responde de manera concisa, profesional y natural
            2. Organiza tus respuestas de manera estructurada cuando sea apropiado
            
            INSTRUCCIÓN CRÍTICA SOBRE INFORMACIÓN INCOMPLETA:
            - Si no encuentras datos concretos sobre lo que se te pregunta, NO INVENTES información.
            - En su lugar, responde de forma natural con frases como:
              • "No encuentro detalles específicos sobre [tema] en el CV, pero puedo compartirte esta información relacionada:"
              • "El CV no menciona [tema] en detalle, pero sí incluye estos datos que podrían interesarte:"
              • "Aunque no hay información específica sobre [tema], el perfil indica lo siguiente:"
            - Usa SOLO la información factual presente en los documentos.
            - Si se te pregunta por experiencia laboral y solo tienes información general sobre años de experiencia sin empresas concretas, mencionalo naturalmente.
            
            Recuerda que tu objetivo es ayudar a entender las cualificaciones, experiencia y aptitudes de estos candidatos
            de la manera más precisa posible, sin añadir información que no esté documentada.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Add conversation history
            for msg in self.conversation:
                messages.append(msg)
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            return messages
        
        elif self.active_model == 'claude':
            # Format for Claude
            system_prompt = f"""
            \n\nHuman: Eres un asistente especializado en análisis de hojas de vida. Estás familiarizado con los perfiles de {people_str}.
            
            Al responder consultas sobre estos candidatos:
            1. Responde de manera concisa, profesional y natural
            2. Organiza tus respuestas de manera estructurada cuando sea apropiado
            
            INSTRUCCIÓN CRÍTICA SOBRE INFORMACIÓN INCOMPLETA:
            - Si no encuentras datos concretos sobre lo que se te pregunta, NO INVENTES información.
            - En su lugar, responde de forma natural con frases como:
              • "No encuentro detalles específicos sobre [tema] en el CV, pero puedo compartirte esta información relacionada:"
              • "El CV no menciona [tema] en detalle, pero sí incluye estos datos que podrían interesarte:"
              • "Aunque no hay información específica sobre [tema], el perfil indica lo siguiente:"
            - Usa SOLO la información factual presente en los documentos.
            - Si se te pregunta por experiencia laboral y solo tienes información general sobre años de experiencia sin empresas concretas, mencionalo naturalmente.
            
            Recuerda que tu objetivo es ayudar a entender las cualificaciones, experiencia y aptitudes de estos candidatos
            de la manera más precisa posible, sin añadir información que no esté documentada.
            """
            
            # Add conversation history
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Add current query
            query_text = f"\n\nHuman: {query}"
            
            # Combine everything
            prompt = system_prompt + conversation_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Simple format for fallback
            return {"query": query}
    
    async def _query_needs_rag(self, query: str) -> bool:
        """
        Determines if a query needs to use RAG or can be answered directly.
        Mejorado para reconocer diversas formas de consultar la misma información.
        
        Args:
            query: User query
            
        Returns:
            True if the query needs RAG, False if not
        """
        # Generic queries that don't need RAG
        generic_greetings = [
            "hola", "hello", "hi", "buenos días", "buenas tardes",
            "cómo estás", "how are you", "saludos", "qué tal",
            "adiós", "bye", "hasta luego", "gracias", "thank you",
        ]
        
        # Check if it's a simple greeting
        if query.lower().strip().strip("?!.,") in generic_greetings:
            logger.info(f"Generic query detected, RAG will not be used: '{query}'")
            return False
        
        # Si hay personas activas, casi siempre usar RAG
        if self.active_people:
            logger.info(f"Active people in conversation ({self.active_people}), RAG will be used: '{query}'")
            return True
        
        # MEJORA: Patrones de consulta comunes que no necesitan contexto
        instruction_patterns = [
            r'\b(explica|dime|describe|qué es)\b.*(sistema|funcionamiento|cómo funciona|utiliza)',
            r'\b(ayuda|ayúdame|instrucciones)\b',
            r'\b(qué puedes hacer|qué haces|cuáles son tus funciones)\b'
        ]
        
        # Verificar si es una consulta de instrucciones sobre el sistema
        for pattern in instruction_patterns:
            if re.search(pattern, query.lower()):
                logger.info(f"System instruction query detected, RAG will not be used: '{query}'")
                return False
        
        # Enhanced keyword list with more specific CV-related terms
        rag_keywords = [
            # Personas
            "pedro", "jorge", "leonardo", 
            
            # Documentos
            "cv", "resume", "curriculum", "hoja de vida", "perfil", "profile",
            
            # Educación
            "educación", "education", "título", "degree", "universidad", "university",
            "estudios", "formación", "academic", "académica", "master", "doctorado",
            "phd", "licenciatura", "diplomado", "curso", "certificate", "certificación",
            
            # Experiencia
            "experiencia", "experience", "trabajo", "job", "profesional", "professional",
            "empresa", "company", "puesto", "position", "cargo", "role", "empleo",
            "trayectoria", "carrera", "career", "historia laboral", "work history",
            
            # Habilidades
            "habilidades", "skills", "tecnología", "technology", "programación", "programming",
            "idiomas", "languages", "competencias", "competencies", "aptitudes", "abilities",
            "conocimientos", "knowledge", "herramientas", "tools", "software", "hardware",
            "frameworks", "metodologías", "methodologies", "especialidad", "specialty",
            
            # Logros
            "logros", "achievements", "proyectos", "projects", "resultados", "results",
            "éxitos", "success", "premios", "awards", "reconocimientos", "recognition",
            
            # Términos técnicos
            "desarrollador", "developer", "ingeniero", "engineer", "frontend", "backend",
            "fullstack", "java", "python", "javascript", "html", "css", "database", "base de datos",
            "cloud", "nube", "agile", "ágil", "devops", "ui", "ux", "testing", "qa",
            
            # Personal
            "contacto", "contact", "referencias", "references", "personal", "edad", "age",
            "dirección", "address", "email", "teléfono", "phone", "disponibilidad", "availability"
        ]
        
        # Check if query contains any CV-related keyword
        query_lower = query.lower()
        for keyword in rag_keywords:
            if keyword.lower() in query_lower:
                logger.info(f"Domain-related query (keyword: {keyword}), RAG will be used: '{query}'")
                return True
        
        # MEJORA: Detectar intención de comparación
        comparison_patterns = [
            r'\b(compar|diferencia|similitud|mejor|peor|vs|versus)\b',
            r'\b(quién|quien|cuál|cual) (es|tiene|posee) (mejor|más|mayor|menor)\b',
            r'\btodos\b.*(tienen|poseen|cuentan)'
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Comparative query detected, RAG will be used: '{query}'")
                self.conversation_state["comparative_mode"] = True
                return True
        
        # For other messages, more sophisticated decision
        # If more than 5 words, could be a complex question
        if len(query.split()) > 5:  
            logger.info(f"Complex query detected, RAG will be used as precaution: '{query}'")
            return True
        
        # By default, don't use RAG
        logger.info(f"Uncategorized query, RAG will not be used: '{query}'")
        return False
    
    async def _generate_response_with_openai(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a response using the OpenAI API.
        
        Args:
            messages: List of messages in OpenAI format
            
        Returns:
            Generated response
        """
        try:
            # Temperatura más baja para reducir invenciones
            temperature = 0.3
            
            # Use GPT-4 for better responses if available, otherwise fallback to GPT-3.5
            model_to_use = "gpt-4" if os.environ.get("USE_GPT4", "false").lower() == "true" else "gpt-3.5-turbo"
            
            response = self.openai_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta de nuevo o reformula tu pregunta."
    
    async def _generate_response_with_claude(self, prompt: str) -> str:
        """
        Generates a response using the Claude API.
        
        Args:
            prompt: Prompt for Claude
            
        Returns:
            Generated response
        """
        try:
            # Temperatura más baja para reducir invenciones
            temperature = 0.3
            
            # Claude 3 Haiku is a good balance of quality and speed
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta de nuevo o reformula tu pregunta."
    
    async def _generate_response_with_rag_only(self, query_data: Dict[str, Any]) -> str:
        """
        Generates a simple response using only the RAG results.
        
        Args:
            query_data: Query data
            
        Returns:
            Generated response
        """
        context = query_data.get("context", "")
        approx = query_data.get("approx_mode", False)
        query = query_data.get("query", "")
        
        # Use target people if available
        target_people = self._detect_person_in_query(query if query else "") or self.active_people
        
        # MEJORA: Usar nombres completos para respuestas más naturales
        if target_people:
            full_names = [self.person_full_names.get(p, p.capitalize()) for p in target_people]
            if len(full_names) == 1:
                people_str = full_names[0]
            elif len(full_names) == 2:
                people_str = f"{full_names[0]} y {full_names[1]}"
            else:
                people_str = ", ".join(full_names[:-1]) + f" y {full_names[-1]}"
        else:
            people_str = "los candidatos"
        
        # Modo aproximado
        if approx:
            summary = self._summarize_chunks(query_data.get("chunks", []))
            return (
                f"No encontré información específica para responder con total certeza sobre {people_str}. "
                "Sin embargo, basándome en la información más relevante disponible, puedo compartirte lo siguiente:\n\n"
                f"{summary if summary else '• No se encontraron detalles relevantes en los CV disponibles.'}"
            )
        
        if not context:
            return (f"No encontré información disponible sobre {people_str} que coincida "
                    "con tu pregunta. Por favor, intenta ser más específico o pregunta por otro aspecto de su perfil.")
        
        # Improved response format for RAG-only mode
        is_comparative = query_data.get("metadata", {}).get("is_comparative", False)
        
        if is_comparative:
            response = f"Comparando los perfiles de {people_str}, puedo compartirte la siguiente información:"
        else:
            response = f"Sobre {people_str}, puedo compartirte la siguiente información:"
        
        # Process context to make it more readable
        paragraphs = context.split("\n\n")
        formatted_paragraphs = []
        
        for para in paragraphs:
            # Clean up paragraph
            para = para.strip()
            if not para:
                continue
                
            # Add paragraph with bullet point
            formatted_paragraphs.append(f"• {para}")
        
        # Join paragraphs
        if formatted_paragraphs:
            response += "\n\n" + "\n\n".join(formatted_paragraphs)
        else:
            response += "\n\n" + context
        
        return response
    
    async def _process_query(self, query: str) -> str:
        """
        Processes a user query and generates a response.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        # Determine if the query needs RAG
        needs_rag = await self._query_needs_rag(query)
        
        # If it needs RAG, perform search
        rag_results = {}
        if needs_rag:
            # Use active people if available for better status message
            target_people = self._detect_person_in_query(query) or self.active_people
            
            # MEJORA: Detectar si es una consulta comparativa
            is_comparative = (
                'comparar' in query.lower() or 
                'compara' in query.lower() or 
                'los tres' in query.lower() or 
                'todos' in query.lower() or
                len(target_people) > 1 or
                self.conversation_state.get("comparative_mode", False)
            )
            
            if is_comparative:
                # Si no hay personas objetivo específicas en una consulta comparativa, usar todos
                if not target_people:
                    target_people = ["pedro", "jorge", "leonardo"]
                    self.active_people = target_people
                print(f"{Colors.DIM}Realizando análisis comparativo de {', '.join(target_people)}...{Colors.RESET}", flush=True)
            elif target_people:
                people_str = ", ".join(target_people)
                print(f"{Colors.DIM}Buscando información específica sobre {people_str}...{Colors.RESET}", flush=True)
            else:
                print(f"{Colors.DIM}Buscando información en las hojas de vida...{Colors.RESET}", flush=True)
                
            rag_results = await self._perform_rag_search(query)
            
            # Verificar si tenemos evidencia "concreta" suficiente
            approx_mode = False
            max_sim = rag_results.get('max_similarity', 0)
            if needs_rag and (not rag_results.get('chunks') or max_sim < MIN_SIMILARITY_CONCRETA):
                approx_mode = True
                logger.info(f"Activando modo aproximado: max_sim={max_sim:.4f} < {MIN_SIMILARITY_CONCRETA}")
                
                # Forzar un TOP-k más grande para intentar capturar algo útil
                fallback_results = await self.search_pipeline.search(
                    query_text=query,
                    mode='hybrid',
                    top_k=FALLBACK_TOP_K,
                    strategy='relevance'
                )
                
                # Asegurar que filtramos por las personas correctas
                if target_people:
                    doc_ids = self._get_person_doc_ids_from_query(query)
                    if doc_ids and fallback_results and fallback_results.get('chunks'):
                        # Filtrar chunks para mantener solo los de las personas objetivo
                        logger.info(f"Filtrando {len(fallback_results.get('chunks', []))} resultados para doc_ids: {doc_ids}")
                        filtered_chunks = []
                        for chunk in fallback_results.get('chunks', []):
                            if chunk.get('doc_id') in doc_ids:
                                filtered_chunks.append(chunk)
                                logger.info(f"Incluyendo chunk de doc_id={chunk.get('doc_id')}")
                            else:
                                logger.info(f"Excluyendo chunk de doc_id={chunk.get('doc_id')} (no coincide)")
                        
                        if filtered_chunks:
                            logger.info(f"Modo aproximado: {len(filtered_chunks)} chunks filtrados por persona")
                            fallback_results['chunks'] = filtered_chunks
                            # Regenerar contexto
                            context = ""
                            for chunk in filtered_chunks:
                                if 'content' in chunk and chunk['content']:
                                    doc_header = f"[Documento: {chunk['title']}]\n"
                                    context += doc_header + chunk['content'] + "\n\n"
                            fallback_results['context'] = context
                        else:
                            logger.warning("Todos los chunks filtrados, sin resultados específicos")
                
                if fallback_results and fallback_results.get('chunks'):
                    rag_results = fallback_results
                    logger.info(f"Modo aproximado: usando {len(fallback_results.get('chunks', []))} resultados de TOP-K={FALLBACK_TOP_K}")
                else:
                    logger.warning("Modo aproximado: sin resultados")
                
                # Registrar resultados para depuración
                logger.info(f"Modo aproximado activado: max_sim={max_sim}, nuevo top_k={FALLBACK_TOP_K}")
            
            # Actualizar el modo aproximado en los resultados
            rag_results['approx_mode'] = approx_mode
            
            # Log de resultados para depuración
            if rag_results:
                chunks = rag_results.get('chunks', [])
                logger.info(f"RAG search returned {len(chunks)} chunks")
                
                # Log primeras líneas de contexto para depuración
                context = rag_results.get('context', '')
                context_preview = context[:500] + "..." if len(context) > 500 else context
                logger.info(f"Context preview: {context_preview}")
        
        # Generate prompt according to active model
        if needs_rag and rag_results:
            # Prompt with RAG context
            prompt = self._create_prompt_with_rag_context(query, rag_results)
        else:
            # Prompt without RAG context
            prompt = self._create_simple_prompt(query)
        
        # Generate response according to active model
        if self.active_model == 'openai':
            print(f"{Colors.DIM}Generando respuesta con OpenAI...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_openai(prompt)
        elif self.active_model == 'claude':
            print(f"{Colors.DIM}Generando respuesta con Claude...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_claude(prompt)
        else:
            # Fallback to simple RAG
            print(f"{Colors.DIM}Generando respuesta básica con la información encontrada...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_rag_only(rag_results)
        
        # Update conversation history
        self.conversation.append({"role": "user", "content": query})
        self.conversation.append({"role": "assistant", "content": response})
        
        # Truncate history if it exceeds the maximum
        if len(self.conversation) > self.max_conversation_history * 2:
            self.conversation = self.conversation[-self.max_conversation_history*2:]
        
        return response
    
    def print_welcome(self):
        """Prints welcome message."""
        title = "Asistente Experto en Hojas de Vida - CV Analysis"
        subtitle = f"Modelo activo: {self.active_model.upper()}"
        
        print("\n" + "=" * self.term_width)
        print(f"{Colors.BOLD}{Colors.GREEN}{title.center(self.term_width)}{Colors.RESET}")
        print(f"{Colors.DIM}{subtitle.center(self.term_width)}{Colors.RESET}")
        print("=" * self.term_width)
        
        print(f"\n{Colors.CYAN}¡Bienvenido al Asistente de Hojas de Vida!{Colors.RESET}")
        print("Puedo ayudarte con información detallada sobre los CV de Pedro, Jorge y Leonardo.")
        print("Pregúntame sobre sus habilidades, experiencia, educación o cualquier otro aspecto de sus perfiles profesionales.")
        
        if self.active_model == 'rag-only':
            print(f"\n{Colors.YELLOW}Aviso: No se detectó ninguna clave API de LLM válida.{Colors.RESET}")
            print(f"{Colors.YELLOW}Solo se utilizará el sistema RAG para las respuestas.{Colors.RESET}")
            print(f"{Colors.YELLOW}Para una experiencia completa, configura OPENAI_API_KEY o ANTHROPIC_API_KEY.{Colors.RESET}")
        
        print("\nEscribe tu pregunta y presiona Enter para comenzar.")
        print("Escribe '/salir' o '/exit' para finalizar la conversación.")
        print("=" * self.term_width + "\n")
    
    def format_response(self, response: str) -> str:
        """
        Formats a response to display in the terminal.
        
        Args:
            response: Response to format
            
        Returns:
            Formatted response
        """
        # Apply line wrapping with improved formatting
        lines = []
        for paragraph in response.split('\n'):
            if not paragraph.strip():
                lines.append("")
                continue
            
            # Handle bullet points and lists specially
            if paragraph.strip().startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # For list items, add less indentation for the wrapped lines
                first_line = paragraph
                wrapped_content = textwrap.wrap(
                    paragraph, 
                    width=self.term_width-8,  # Less width to account for indentation
                    subsequent_indent='      '  # Indent subsequent lines
                )
                
                # Replace the first line with the original (keeping the bullet)
                if wrapped_content:
                    lines.append(wrapped_content[0])
                    lines.extend(wrapped_content[1:])
                else:
                    lines.append(paragraph)
            else:
                # Regular paragraph wrapping
                wrapped = textwrap.wrap(paragraph, width=self.term_width-4)
                lines.extend(wrapped)
        
        # Join with indentation and add emphasis to key elements
        formatted_text = ""
        for line in lines:
            indented_line = "    " + line
            
            # Add emphasis to section headers (all caps or ending with colon)
            if line.isupper() or (line.strip().endswith(':') and len(line.strip()) < 30):
                indented_line = f"{Colors.BOLD}{indented_line}{Colors.RESET}"
            
            # Highlight candidate names with more variants
            for person, full_name in self.person_full_names.items():
                name_parts = full_name.split()
                for name_part in name_parts:
                    if name_part in indented_line:
                        indented_line = indented_line.replace(
                            name_part, f"{Colors.BOLD}{Colors.CYAN}{name_part}{Colors.RESET}"
                        )
            
            # Highlight important keywords
            keywords = {
                "experiencia": f"{Colors.GREEN}experiencia{Colors.RESET}",
                "educación": f"{Colors.GREEN}educación{Colors.RESET}",
                "formación": f"{Colors.GREEN}formación{Colors.RESET}",
                "habilidades": f"{Colors.GREEN}habilidades{Colors.RESET}",
                "competencias": f"{Colors.GREEN}competencias{Colors.RESET}",
                "proyectos": f"{Colors.GREEN}proyectos{Colors.RESET}",
                "logros": f"{Colors.GREEN}logros{Colors.RESET}"
            }
            
            for keyword, highlighted in keywords.items():
                # Replace only whole words
                indented_line = re.sub(
                    r'\b' + keyword + r'\b', 
                    highlighted, 
                    indented_line, 
                    flags=re.IGNORECASE
                )
            
            formatted_text += indented_line + "\n"
        
        return formatted_text
    
    async def run(self):
        """Runs the interactive chat."""
        # Initialize components
        if not await self.initialize():
            print(f"\n{Colors.RED}Error al inicializar el sistema. Consulta los registros para más detalles.{Colors.RESET}\n")
            return
        
        # Show welcome
        self.print_welcome()
        
        # Main loop
        try:
            while True:
                # Update terminal size
                self.term_width, self.term_height = shutil.get_terminal_size()
                
                # Request input
                query = input(f"\n{Colors.BOLD}{Colors.GREEN}Tú:{Colors.RESET} ")
                
                # Check if empty
                if not query.strip():
                    continue
                
                # Check if exit command
                if query.lower() in ['/salir', '/exit', '/quit']:
                    print(f"\n{Colors.GREEN}¡Gracias por usar el Asistente de Hojas de Vida! ¡Hasta pronto!{Colors.RESET}\n")
                    break
                
                # Check for diagnostics command
                if query.lower() in ['/diagnostico', '/diagnose', '/debug']:
                    print(f"\n{Colors.YELLOW}Ejecutando diagnóstico completo...{Colors.RESET}")
                    await self.run_diagnostics()
                    print(f"\n{Colors.YELLOW}Diagnóstico completado. Revisa los logs para más detalles.{Colors.RESET}")
                    continue
                
                # Process query
                response = await self._process_query(query)
                
                # Show response
                print(f"\n{Colors.BOLD}{Colors.BLUE}Asistente:{Colors.RESET}")
                formatted_response = self.format_response(response)
                print(formatted_response)
                
                # Mostrar estado de conversación (para depuración)
                if os.environ.get("DEBUG_CONVERSATION", "false").lower() == "true":
                    print(f"\n{Colors.DIM}[DEBUG] Personas activas: {self.active_people}{Colors.RESET}")
                    print(f"{Colors.DIM}[DEBUG] Último tema: {self.conversation_state.get('last_topic', '')}{Colors.RESET}")
                    print(f"{Colors.DIM}[DEBUG] Modo comparativo: {self.conversation_state.get('comparative_mode', False)}{Colors.RESET}")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n{Colors.RED}Error inesperado: {e}{Colors.RESET}\n")
        finally:
            # Close resources
            await self.close()

async def main():
    """Main function for execution from the command line."""
    parser = argparse.ArgumentParser(description='Enhanced CV Assistant - Resume Analysis')
    
    parser.add_argument('--model', choices=['openai', 'claude'],
                      default='openai', help='Modelo a utilizar')
    parser.add_argument('--top-k', type=int, default=5,
                      help='Número de resultados a retornar del RAG')
    parser.add_argument('--use-container-name', action='store_true',
                      help='Usar postgres_pgvector en lugar de localhost')
    parser.add_argument('--debug', action='store_true',
                      help='Activar registro de depuración')
    parser.add_argument('--diagnose-only', action='store_true',
                      help='Solo ejecutar diagnóstico sin iniciar el chat')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app").setLevel(logging.DEBUG)
    
    # Initialize chat
    chat = EnhancedCVAssistant(
        model=args.model,
        top_k=args.top_k,
        use_localhost=not args.use_container_name
    )
    
    # Run diagnostics only if specified
    if args.diagnose_only:
        # Initialize components for diagnostics
        if await chat.initialize():
            print(f"{Colors.YELLOW}Ejecutando diagnóstico completo...{Colors.RESET}")
            await chat.run_diagnostics()
            print(f"{Colors.YELLOW}Diagnóstico completado. Revisa los logs para más detalles.{Colors.RESET}")
            # Close resources
            await chat.close()
            return
    
    # Run chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)