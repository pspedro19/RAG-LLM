# Import all the existing code from agent_tp3.py
import os
import time
import uuid
import asyncio
import logging
import json
import sys
from typing import Dict, List, Set, Any, Optional, Literal, TypedDict, Union, cast
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

import psycopg2
from langgraph.graph import StateGraph, END
import numpy as np
from openai import AsyncOpenAI

# Validación centralizada de entorno (sin salir del programa)
def validate_env(required_vars: list[str]) -> bool:
    """Valida que todas las variables de entorno requeridas estén definidas"""
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print("\n WARNING: Faltan variables de entorno requeridas:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nAlgunas funcionalidades pueden no estar disponibles.")
        return False

    return True

# Carga de variables de entorno
try:
    from dotenv import load_dotenv
    # Intentar cargar desde diferentes ubicaciones posibles
    load_dotenv()  # Busca .env en el directorio actual
    load_dotenv("/root/RAG-LLM/.env")  # Busca en la raíz del proyecto

    # Verificar variables críticas de forma centralizada (solo warning)
    critical_vars = ["OPENAI_API_KEY"]
    validate_env(critical_vars)

    # Mostrar información de configuración
    print("\n Configuración actual:")
    print(f"  DB Host: {os.environ.get('PG_HOST', 'localhost')}")
    print(f"  DB Name: {os.environ.get('POSTGRES_DB', os.environ.get('PG_DATABASE', 'mydatabase'))}")
    print(f"  OpenAI API: {'configurada ' if os.environ.get('OPENAI_API_KEY') else 'no configurada '}")
    print(f"  Tavily API: {'configurada ' if os.environ.get('TAVILY_API_KEY') else 'no configurada '}")

except ImportError:
    print(" python-dotenv no está instalado. Variables de entorno limitadas a las del sistema.")
    validate_env(["OPENAI_API_KEY"])
except Exception as e:
    print(f" Error cargando variables de entorno: {e}")
    validate_env(["OPENAI_API_KEY"])

# Implementación simple de checkpoint manager
class SimpleCheckpointManager:
    """Implementación mejorada de un checkpoint manager con persistencia local"""
    
    def __init__(self, checkpoint_dir=None):
        self.checkpoints = {}
        self.checkpoint_dir = checkpoint_dir
        # Crear directorio si no existe
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir)
                print(f"Directorio de checkpoints creado: {checkpoint_dir}")
            except Exception as e:
                print(f"No se pudo crear directorio de checkpoints: {e}")
    
    def get(self, key, default=None):
        """Recupera un checkpoint por su clave, intentando primero desde memoria,
        luego desde disco si existe."""
        if key in self.checkpoints:
            return self.checkpoints.get(key)
        
        # Intentar cargar desde disco si no está en memoria
        if self.checkpoint_dir:
            try:
                file_path = os.path.join(self.checkpoint_dir, f"{key}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.checkpoints[key] = data  # Cargar a memoria
                        return data
            except Exception as e:
                logging.warning(f"Error cargando checkpoint desde disco: {e}")
        
        return default
    
    def put(self, key, value):
        """Guarda un checkpoint con la clave especificada, tanto en memoria como en disco"""
        self.checkpoints[key] = value
        # Intenta guardar a disco si se especificó un directorio
        if self.checkpoint_dir:
            try:
                file_path = os.path.join(self.checkpoint_dir, f"{key}.json")
                with open(file_path, 'w') as f:
                    json.dump(value, f)
            except Exception as e:
                logging.warning(f"Error guardando checkpoint a disco: {e}")
        return True
    
    def delete(self, key):
        """Elimina un checkpoint"""
        if key in self.checkpoints:
            del self.checkpoints[key]
            
            # Eliminar de disco también
            if self.checkpoint_dir:
                try:
                    file_path = os.path.join(self.checkpoint_dir, f"{key}.json")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.warning(f"Error eliminando checkpoint de disco: {e}")
            
            return True
        return False

# Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("curacao_assistant.log")
    ]
)
logger = logging.getLogger("curacao-assistant")

# Importar servicio LLM unificado
try:
    # Importar desde app/ si está dentro del paquete
    from llm_service import get_llm_service
except ImportError:
    # Importar desde app.llm_service si se ejecuta desde otro contexto
    try:
        from app.llm_service import get_llm_service
    except ImportError:
        # Fallback: importar directamente si está en el mismo directorio
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from llm_service import get_llm_service

# Inicializar servicio LLM con soporte multi-proveedor
llm_service = get_llm_service()
logger.info(f"Servicio LLM inicializado - Proveedores disponibles: {llm_service.get_available_providers()}")

# Mantener referencia al cliente OpenAI para compatibilidad con código legacy
# (será removido gradualmente)
from openai import AsyncOpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    client = AsyncOpenAI(api_key=openai_api_key)
else:
    client = None
    logger.warning("Cliente OpenAI no inicializado (usando solo Claude)")

# Enums y constantes (mejora tipado)
class AgentType(str, Enum):
    CLASSIFIER = "classifier"
    RAG = "rag_agent"
    WEB_SEARCH = "web_search_agent"
    ITINERARY = "itinerary_agent"
    CONVERSATIONAL = "conversational_agent"

class QueryType(str, Enum):
    CONVERSATIONAL = "conversacional"
    INFORMATION = "informacion"
    ITINERARY = "itinerario"

# Modelos Pydantic para validación de estructura (LG-2)
class TokenStats(TypedDict):
    prompt: int
    completion: int
    reasoning: int
    total: Optional[int]

class AgentStats(TypedDict):
    time: float
    tokens: TokenStats
    success: bool
    error: Optional[str]

class ReActStep(BaseModel):
    agent: str
    thought: str
    action: str
    observation: str
    timestamp: float = Field(default_factory=time.time)

    def model_dump(self):
        """Reemplaza dict() para compatibilidad con Pydantic v2"""
        return {
            "agent": self.agent,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "timestamp": self.timestamp
        }

class UserPreferences(BaseModel):
    duracion: Optional[str] = None
    presupuesto: Optional[str] = None
    intereses: List[str] = Field(default_factory=list)
    tipo_viajero: Optional[str] = None
    temporada: Optional[str] = None
    alojamiento: Optional[str] = None

# Estado global tipado (LG-2)
class AssistantState(TypedDict):
    # Datos de entrada
    query: str
    conversation_id: str
    timestamp: float
    
    # Estado de ejecución
    active_agents: Set[str]
    query_type: str
    current_step: int
    max_steps: int
    
    # Resultados de agentes
    rag_results: Dict[str, Any]
    search_results: Dict[str, Any]
    itinerary_plan: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # Historial y memoria
    conversation_history: List[Dict[str, str]]
    react_steps: List[Dict[str, Any]]
    
    # Salida y métricas
    final_response: str
    processing_stats: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

# Implementación de límites y guardrails (MA-8)
MAX_STEPS = 15
MAX_TOKENS = 12000
TIMEOUT_SECONDS = 150 # Aumentado a 150 segundos

# Circuit Breaker base (elimina duplicación)
class CircuitBreakerService:
    """Implementación base del patrón Circuit Breaker para servicios externos"""
    
    def __init__(self, service_name: str, max_failures: int = 3, retry_seconds: int = 60):
        self.service_name = service_name
        self.failure_count = 0
        self.circuit_open = False
        self.circuit_reset_time = None
        self.max_failures = max_failures
        self.base_retry_seconds = retry_seconds
        self.fallback_mode = False
    
    async def _calculate_backoff(self) -> int:
        """Implementa backoff exponencial para reintentos"""
        return self.base_retry_seconds * (2 ** min(self.failure_count, 6))  # Max ~10 min backoff
    
    async def _check_circuit_breaker(self) -> bool:
        """Verifica si el circuit breaker permite operación"""
        if not self.circuit_open:
            return True
        
        current_time = time.time()
        if self.circuit_reset_time and current_time > self.circuit_reset_time:
            logger.info(f"Circuit breaker: intentando restablecer {self.service_name}")
            self.circuit_open = False
            self.failure_count = 0
            return True
        
        return False
    
    def _register_failure(self, error: Exception, trace_id: str = None) -> None:
        """Registra un fallo y activa el circuit breaker si es necesario"""
        self.failure_count += 1
        log_prefix = f"[{trace_id}] " if trace_id else ""
        logger.error(f"{log_prefix}Error en {self.service_name} ({self.failure_count}/{self.max_failures}): {error}")
        
        if self.failure_count >= self.max_failures:
            asyncio.create_task(self._activate_circuit_breaker(trace_id))
    
    async def _activate_circuit_breaker(self, trace_id: str = None) -> None:
        """Activa el circuit breaker y calcula el tiempo de espera"""
        backoff_time = await self._calculate_backoff()
        self.circuit_open = True
        self.circuit_reset_time = time.time() + backoff_time
        log_prefix = f"[{trace_id}] " if trace_id else ""
        logger.warning(f"{log_prefix}Circuit breaker activado para {self.service_name}. Retry en {backoff_time}s")
        self.fallback_mode = True
    
    def _register_success(self) -> None:
        """Registra una operación exitosa y restablece el contador de fallos"""
        self.failure_count = 0

# Servicio de memoria vectorial mejorado
class VectorMemoryService(CircuitBreakerService):
    """Servicio de memoria vectorial adaptado a la estructura real de la base de datos"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):  # Usar miniLM en lugar de OpenAI
        super().__init__(service_name="Vector Database")
        self.db_config = {
            "host": os.environ.get("PG_HOST", "postgres"),  # Cambiado de "localhost" a "postgres"
            "database": os.environ.get("POSTGRES_DB", os.environ.get("PG_DATABASE", "mydatabase")),
            "user": os.environ.get("POSTGRES_USER", os.environ.get("PG_USER", "myuser")),
            "password": os.environ.get("POSTGRES_PASSWORD", os.environ.get("PG_PASSWORD", "mypassword")),
            "port": int(os.environ.get("PG_PORT", 5432))
        }
        self.conn = None
        self.llm_service = llm_service
        self.embedding_model = embedding_model  # Almacenamos el nombre pero no lo usamos con OpenAI
        self.top_k = 8  # Aumentado a 8 para mayor contexto
        
        # Importar localmente para evitar dependencias innecesarias en todo el código
        try:
            from sentence_transformers import SentenceTransformer
            # Cargar el modelo de embeddings local
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"Modelo de embedding local cargado: {embedding_model}")
        except (ImportError, RuntimeError, OSError) as e:
            # Manejar errores de importación, incluyendo problemas de bitsandbytes en Windows
            logger.warning(f"No se pudo cargar SentenceTransformer localmente: {str(e)[:100]}")
            logger.info("Usando OpenAI Embeddings API como fallback")
            self.model = None
            self.use_openai_embeddings = True
    
    async def initialize(self):
        """Inicializa la conexión a la base de datos con circuit breaker"""
        if self.fallback_mode or not await self._check_circuit_breaker():
            return False
            
        try:
            self.conn = psycopg2.connect(**self.db_config)
            cursor = self.conn.cursor()
            # Verificar conexión con consulta simple
            cursor.execute("SELECT 1")
            if cursor.fetchone()[0] == 1:
                logger.info(f"Conexión exitosa a PostgreSQL en {self.db_config['host']}")
                self._register_success()
                return True
            else:
                raise Exception("La conexión no responde correctamente")
                
        except Exception as e:
            self._register_failure(e)
            return False
    
    async def generate_embedding(self, text):
        """Genera embedding usando el modelo local o OpenAI API como fallback"""
        start_time = time.time()
        try:
            if self.model is None:
                # Fallback a OpenAI Embeddings API (solo si está disponible)
                if self.llm_service.openai_client:
                    logger.info("Usando OpenAI Embeddings API (fallback)")
                    embedding_list = await self.llm_service.embeddings(text)
                    logger.info(f"Embedding generado con OpenAI API en {time.time() - start_time:.2f}s")
                    return embedding_list, {"tokens": len(text.split())}
                else:
                    # Si no hay OpenAI, usar embeddings aleatorios normalizados
                    logger.warning("No hay OpenAI disponible para embeddings, usando vector aleatorio")
                    random_embedding = np.random.normal(0, 1, 384)
                    normalized = random_embedding / np.linalg.norm(random_embedding)
                    return normalized.tolist(), 0

            # Generar embedding con SentenceTransformer (llamada síncrona)
            embedding = self.model.encode([text])[0]

            # Convertir a lista y normalizar si es necesario
            embedding_list = embedding.tolist()
            
            # Estimar tokens (aproximado)
            prompt_tokens = len(text.split()) 
            
            logger.info(f"Embedding generado localmente: {len(embedding_list)} dimensiones, tiempo: {time.time()-start_time:.2f}s")
            return embedding_list, prompt_tokens
            
        except Exception as e:
            logger.error(f"Error generando embedding local: {e}")
            # Fallback: vector aleatorio normalizado para evitar fallo completo
            # Dimensión de 384 para MiniLM
            random_embedding = np.random.normal(0, 1, 384)
            normalized = random_embedding / np.linalg.norm(random_embedding)
            return normalized.tolist(), 0
    
    async def search(self, query: str, trace_id: str) -> Dict[str, Any]:
        """Busca información relevante adaptado a la estructura real de la base de datos"""
        logger.info(f"[{trace_id}] Realizando búsqueda vectorial: '{query[:50]}...'")
        start_time = time.time()
        
        # Comprobar circuit breaker y fallback
        if self.fallback_mode or self.circuit_open:
            logger.warning(f"[{trace_id}] Usando modo fallback para búsqueda vectorial")
            return {
                "chunks": [],
                "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                "fallback": True,
                "time": time.time() - start_time
            }
        
        if not self.conn:
            success = await self.initialize()
            if not success:
                self.fallback_mode = True
                return {
                    "chunks": [],
                    "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                    "fallback": True,
                    "time": time.time() - start_time
                }
        
        try:
            # Generar embedding para la consulta
            query_embedding, prompt_tokens = await self.generate_embedding(query)
            
            # Obtener el modelo usado en la base de datos para verificar
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT model_name FROM embeddings LIMIT 1")
            db_model = cursor.fetchone()
            if db_model:
                logger.info(f"Modelo en base de datos: {db_model[0]}")
            
            # Consulta adaptada a la estructura real de las tablas
            # Usamos el operador <=> de pgvector para calcular la similitud
            sql = """
            SELECT 
                c.chunk_id, c.content, d.title, d.doc_id,
                1 - (e.embedding <=> %s::vector) AS cosine_similarity
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE 1 - (e.embedding <=> %s::vector) > 0.65
            ORDER BY cosine_similarity DESC
            LIMIT %s
            """
            
            # También crear una consulta de respaldo por palabras clave
            keyword_sql = """
            SELECT 
                c.chunk_id, c.content, d.title, d.doc_id,
                0.7 AS cosine_similarity
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE 
                c.content ILIKE %s
                AND c.chunk_id NOT IN (
                    SELECT c2.chunk_id
                    FROM embeddings e2
                    JOIN chunks c2 ON e2.chunk_id = c2.chunk_id
                    JOIN documents d2 ON c2.doc_id = d2.doc_id
                    WHERE 1 - (e2.embedding <=> %s::vector) > 0.65
                    ORDER BY 1 - (e2.embedding <=> %s::vector) DESC
                    LIMIT %s
                )
            LIMIT 3
            """
            
            # Ejecutar consulta vectorial
            cursor.execute(sql, (query_embedding, query_embedding, self.top_k))
            vector_results = cursor.fetchall()
            
            # Si hay pocos resultados vectoriales, complementar con búsqueda por palabras clave
            keyword_results = []
            if len(vector_results) < self.top_k:
                cursor.execute(
                    keyword_sql, 
                    (f"%{query}%", query_embedding, query_embedding, self.top_k)
                )
                keyword_results = cursor.fetchall()
            
            # Combinar resultados
            all_results = vector_results + keyword_results
            
            # Procesar resultados (sin intentar acceder a columnas que no existen)
            chunks = []
            for result in all_results:
                chunk_id, content, title, doc_id, similarity = result
                chunks.append({
                    'title': title,
                    'content': content,
                    'relevance': float(similarity),
                    'source': doc_id
                })
            
            # Registrar éxito
            self._register_success()
            
            processing_time = time.time() - start_time
            logger.info(f"[{trace_id}] Búsqueda vectorial completada: {len(chunks)} resultados en {processing_time:.2f}s")
            
            return {
                "chunks": chunks,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": 0,
                    "reasoning": prompt_tokens
                },
                "time": processing_time,
                "fallback": False
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._register_failure(e, trace_id)
            logger.error(f"[{trace_id}] Error en búsqueda vectorial: {e}")
            
            # Intentar búsqueda de fallback simple si falla la vectorial
            try:
                if self.conn:
                    cursor = self.conn.cursor()
                    fallback_sql = """
                    SELECT c.chunk_id, c.content, d.title, d.doc_id, 0.7 AS similarity
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.content ILIKE %s
                    ORDER BY RANDOM()
                    LIMIT %s
                    """
                    
                    cursor.execute(fallback_sql, (f"%{query}%", self.top_k))
                    results = cursor.fetchall()
                    
                    chunks = []
                    for result in results:
                        chunk_id, content, title, doc_id, similarity = result
                        chunks.append({
                            'title': title,
                            'content': content,
                            'relevance': float(similarity),
                            'source': doc_id
                        })
                    
                    if chunks:
                        logger.info(f"[{trace_id}] Búsqueda de respaldo por palabras clave exitosa: {len(chunks)} resultados")
                        return {
                            "chunks": chunks,
                            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                            "time": time.time() - start_time,
                            "fallback": True
                        }
            except Exception as fallback_error:
                logger.error(f"[{trace_id}] Error en búsqueda de respaldo: {fallback_error}")
                # Si también falla el fallback, continuar al return por defecto
            
            return {
                "chunks": [],
                "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                "error": str(e),
                "fallback": True,
                "time": processing_time
            }

# Servicio de búsqueda web mejorado
class WebSearchService(CircuitBreakerService):
    """Servicio de búsqueda web usando Tavily con fallback a LLM"""
    
    def __init__(self):
        super().__init__(service_name="Web Search")
        self.api_key = os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("No se encontró API key para Tavily. El servicio usará el modo fallback.")
            self.fallback_mode = True
    
    async def search(self, query: str, trace_id: str) -> Dict[str, Any]:
        """Realiza búsqueda web con Tavily API"""
        try:
            import requests
        except ImportError:
            logger.error("No se pudo importar 'requests'. Usando fallback para búsqueda web.")
            return await self._fallback_search(query, trace_id)
        
        logger.info(f"[{trace_id}] Iniciando búsqueda con Tavily API: '{query[:50]}...'")
        start_time = time.time()
        
        # Verificar circuit breaker y fallback
        if self.fallback_mode or not await self._check_circuit_breaker():
            logger.warning(f"[{trace_id}] Usando modo fallback para búsqueda web")
            return await self._fallback_search(query, trace_id)
        
        # Implementa timeout para evitar bloqueos largos
        try:
            tavily_url = "https://api.tavily.com/search"
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Curazao incluido en query si falta
            search_query = f"Curazao {query}" if "curazao" not in query.lower() else query
            
            # Payload optimizado para resultados más relevantes
            payload = {
                "query": search_query,
                "search_depth": "advanced",  # Búsqueda más profunda
                "max_results": 8,  # Aumentado para más contexto
                "include_answer": True,
                "include_raw_content": False,  # Optimización para reducir tamaño
                "include_domains": [
                    "tripadvisor.com", 
                    "lonelyplanet.com", 
                    "curacao.com", 
                    "visitcuracao.com",
                    "visitaruba.com",
                    "caribbeancruisecompany.com",
                    "wikitravel.org",
                    "wikivoyage.org"
                ]
            }
            
            # Implementar timeout para evitar bloqueos
            response = requests.post(
                tavily_url, 
                headers=headers, 
                json=payload,
                timeout=30  # Incrementado para búsquedas profundas
            )
            response.raise_for_status()
            
            search_data = response.json()
            processing_time = time.time() - start_time
            
            # Extraer y formatear resultados con mejor estructura
            formatted_results = ""
            if "answer" in search_data and search_data["answer"]:
                formatted_results += f"### Resumen\n{search_data['answer']}\n\n"
                
            for result in search_data.get("results", []):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                
                formatted_results += f"### {title}\nFuente: {url}\n{content}\n\n"
            
            # Registrar éxito
            self._register_success()
            
            # Estimación de tokens (aproximada)
            prompt_tokens_estimate = len(query) // 4
            completion_tokens_estimate = len(formatted_results) // 4
            
            logger.info(f"[{trace_id}] Búsqueda Tavily exitosa: {len(search_data.get('results', []))} resultados en {processing_time:.2f}s")
            
            return {
                "formatted": formatted_results,
                "raw": search_data.get("results", []),
                "time": processing_time,
                "tokens": {
                    "prompt": prompt_tokens_estimate,
                    "completion": completion_tokens_estimate,
                    "reasoning": prompt_tokens_estimate
                },
                "fallback": False
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._register_failure(e, trace_id)
            logger.error(f"[{trace_id}] Error en búsqueda Tavily: {e}")
            
            # Usar fallback LLM
            return await self._fallback_search(query, trace_id)
    
    async def _fallback_search(self, query: str, trace_id: str) -> Dict[str, Any]:
        """Sistema de fallback para búsqueda web usando LLM"""
        start_time = time.time()
        logger.info(f"[{trace_id}] Ejecutando búsqueda web con LLM de respaldo")
        
        try:
            # Prompt mejorado para simular búsqueda web
            fallback_prompt = f"""
            Actúa como si fueras un servicio de búsqueda web especializado en turismo en Curazao.
            La consulta original es: "{query}"
            
            Genera 4 resultados detallados de búsqueda que podrían provenir de sitios web turísticos confiables sobre Curazao.
            Cada resultado debe tener:
            - Un título realista como si fuera de una página web turística
            - Una URL ficticia pero plausible (como visitcuracao.com, tripadvisor.com/curacao, etc.)
            - Un párrafo detallado con información relevante que responda directamente a la consulta
            
            También genera un resumen general bien estructurado como si fuera una respuesta sintetizada de los resultados.
            
            IMPORTANTE: 
            - La información debe ser precisa sobre Curazao
            - Incluye datos específicos cuando sea apropiado (nombres de lugares, horarios, precios aproximados)
            - Si no estás seguro de algún dato específico, proporciona información general correcta
            - Adapta las respuestas para que sean directamente relevantes a la consulta
            """
            
            response = await llm_service.chat_completion(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Eres un sistema de búsqueda web especializado en turismo en Curazao con información actualizada y precisa."},
                    {"role": "user", "content": fallback_prompt}
                ],
                temperature=0.7,
            )

            content = response["content"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            
            processing_time = time.time() - start_time
            logger.info(f"[{trace_id}] Búsqueda LLM fallback completada en {processing_time:.2f}s")
            
            return {
                "formatted": content,
                "raw": [],  # No hay resultados raw en fallback
                "time": processing_time,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "reasoning": prompt_tokens
                },
                "fallback": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[{trace_id}] Error en búsqueda LLM fallback: {e}")
            
            # Respuesta mínima en caso de error total
            return {
                "formatted": "No se pudo obtener información turística sobre Curazao en este momento.",
                "raw": [],
                "time": processing_time,
                "error": str(e),
                "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                "fallback": True
            }

# Inicializar servicios como singletons
vector_service = VectorMemoryService()
web_service = WebSearchService()

# Inicializar servicios de notificación
try:
    from notifications import EmailService, WebhookService
    email_service = EmailService(fallback_mode=True)
    webhook_service = WebhookService()
    logger.info(" Servicios de notificación inicializados")
except ImportError as e:
    logger.warning(f"️ Servicios de notificación no disponibles: {e}")
    email_service = None
    webhook_service = None

# Implementación de nodos con SRP (LG-1)
async def query_classifier(state: AssistantState) -> AssistantState:
    """
    Nodo clasificador: Analiza la consulta y determina el tipo y agentes a activar.
    Single Responsibility: Clasificación y enrutamiento inicial.
    """
    trace_id = state["conversation_id"]
    start_time = time.time()
    logger.info(f"[{trace_id}] Clasificando consulta: '{state['query'][:50]}...'")
    
    # Prompt para clasificador 
    classifier_prompt = f"""
    Analiza la siguiente consulta sobre Curazao. Tu tarea es clasificarla en uno de estos tipos:
    
    1. "conversacional": Saludos simples, charla social o consultas muy básicas que no requieren búsqueda.
       Ejemplos: "hola", "cómo estás", "gracias por la información"
    
    2. "informacion": Preguntas concretas sobre datos o hechos de Curazao.
       Ejemplos: "¿cuáles son las mejores playas?", "clima en abril", "costo de vida"
    
    3. "itinerario": Solicitudes de planificación de viaje o itinerarios personalizados.
       Ejemplos: "planea mi viaje de 5 días", "itinerario para familia con niños", "qué hacer en 3 días"
    
    Consulta: "{state['query']}"
    
    {json.dumps({"format": "Responde solo con una de estas tres palabras: conversacional, informacion o itinerario"})}
    """
    
    # Clasificar consulta con structured output controlado
    try:
        response = await llm_service.chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un clasificador preciso de consultas turísticas. Categoriza exactamente según las instrucciones."},
                {"role": "user", "content": classifier_prompt}
            ],
            temperature=0.1,
            max_tokens=20  # Limitado para forzar respuesta concisa
        )

        # Extraer respuesta y normalizarla
        query_type = response["content"].strip().lower()
        
        # Validación estricta (guardrail)
        if query_type not in ["conversacional", "informacion", "itinerario"]:
            logger.warning(f"[{trace_id}] Clasificación inválida '{query_type}', usando 'conversacional' por defecto")
            query_type = "conversacional"
        
        # Determinar agentes a activar según clasificación
        active_agents = set()
        
        if query_type == "conversacional":
            active_agents.add(AgentType.CONVERSATIONAL.value)
        elif query_type == "informacion":
            active_agents.add(AgentType.RAG.value)
            active_agents.add(AgentType.WEB_SEARCH.value)
            active_agents.add(AgentType.CONVERSATIONAL.value)
        else:  # itinerario
            active_agents.add(AgentType.RAG.value)
            active_agents.add(AgentType.WEB_SEARCH.value)
            active_agents.add(AgentType.ITINERARY.value)
            active_agents.add(AgentType.CONVERSATIONAL.value)
        
        # ReAct step para observabilidad
        react_step = ReActStep(
            agent=AgentType.CLASSIFIER.value,
            thought=f"Analizando la consulta: '{state['query']}' para determinar su tipo...",
            action="Clasificar consulta según patrones lingüísticos",
            observation=f"La consulta se clasifica como '{query_type}'. Activando agentes: {', '.join(active_agents)}."
        )
        
        # Stats para observabilidad
        processing_time = time.time() - start_time

        # Actualizar estado con toda la información
        state["query_type"] = query_type
        state["active_agents"] = active_agents
        state["react_steps"] = [react_step.model_dump()]  # Usar model_dump() en lugar de dict()
        state["current_step"] = 1
        state["processing_stats"] = {
            AgentType.CLASSIFIER.value: {
                "time": processing_time,
                "tokens": {
                    "prompt": response["usage"]["prompt_tokens"],
                    "completion": response["usage"]["completion_tokens"],
                    "reasoning": response["usage"]["prompt_tokens"]
                },
                "success": True,
                "provider": response["provider"]
            }
        }
        
        logger.info(f"[{trace_id}] Consulta clasificada como '{query_type}' en {processing_time:.2f}s")
        
    except Exception as e:
        # Manejo de errores con fallback
        logger.error(f"[{trace_id}] Error en clasificador: {e}")
        
        # Valores por defecto en caso de error
        state["query_type"] = "conversacional"
        state["active_agents"] = {AgentType.CONVERSATIONAL.value}
        state["react_steps"] = [{
            "agent": AgentType.CLASSIFIER.value,
            "thought": "Intentando clasificar la consulta...",
            "action": "Clasificación de consulta",
            "observation": f"Error en clasificación: {str(e)}. Usando tipo 'conversacional' por defecto.",
            "timestamp": time.time()
        }]
        state["errors"] = [f"Error en clasificador: {str(e)}"]
        state["processing_stats"] = {
            AgentType.CLASSIFIER.value: {
                "time": time.time() - start_time,
                "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
                "success": False,
                "error": str(e)
            }
        }
    
    return state

async def rag_agent(state: AssistantState) -> AssistantState:
    """
    Nodo RAG: Consulta información usando base de datos vectorial.
    Single Responsibility: Recuperación de información desde fuente local.
    """
    # Solo ejecutar si está en la lista de agentes activos
    if AgentType.RAG.value not in state["active_agents"]:
        return state
    
    trace_id = state["conversation_id"]
    start_time = time.time()
    logger.info(f"[{trace_id}] Ejecutando agente RAG")
    
    # Incrementar contador de pasos
    state["current_step"] += 1
    
    # Verificar límites de pasos (MA-8)
    if state["current_step"] > state["max_steps"]:
        logger.warning(f"[{trace_id}] Se alcanzó el límite máximo de pasos ({state['max_steps']})")
        state["warnings"].append(f"Flujo truncado por exceder máximo de pasos")
        return state
    
    # Razonamiento ReAct
    thought = f"Necesito buscar información sobre '{state['query']}' en la base de conocimiento vectorial."
    action = "Consultar base de datos vectorial con embeddings semánticos"
    
    # Registrar paso ReAct
    react_step = {
        "agent": AgentType.RAG.value,
        "thought": thought,
        "action": action,
        "observation": "Buscando información relevante...",
        "timestamp": time.time()
    }
    state["react_steps"].append(react_step)
    
    try:
        # Realizar búsqueda vectorial
        rag_results = await vector_service.search(state["query"], trace_id)
        processing_time = time.time() - start_time
        
        # Guardar resultados y actualizar estadísticas
        state["rag_results"] = rag_results
        state["processing_stats"][AgentType.RAG.value] = {
            "time": processing_time,
            "tokens": rag_results.get("tokens", {"prompt": 0, "completion": 0, "reasoning": 0}),
            "success": not rag_results.get("fallback", False) and "error" not in rag_results
        }
        
        if "error" in rag_results:
            state["processing_stats"][AgentType.RAG.value]["error"] = rag_results["error"]
            state["errors"].append(f"Error en RAG: {rag_results['error']}")
        
        # Actualizar observación en ReAct para transparencia
        chunks_count = len(rag_results.get("chunks", []))
        observation = f"Encontré {chunks_count} fragmentos relevantes"
        
        if rag_results.get("fallback", False):
            observation += " (usando sistema de respaldo)"
        
        state["react_steps"][-1]["observation"] = observation
        
        logger.info(f"[{trace_id}] Consulta RAG completada: {chunks_count} resultados en {processing_time:.2f}s")
        
    except Exception as e:
        # Manejo de errores explícito
        processing_time = time.time() - start_time
        error_msg = f"Error en agente RAG: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}")
        
        state["errors"].append(error_msg)
        state["react_steps"][-1]["observation"] = f"Error: {str(e)}. No se pudo recuperar información local."
        state["processing_stats"][AgentType.RAG.value] = {
            "time": processing_time,
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "success": False,
            "error": str(e)
        }
        
        # Fallback: dejar rag_results vacío pero con estructura
        state["rag_results"] = {
            "chunks": [],
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "fallback": True,
            "error": str(e)
        }
    
    return state

async def web_search_agent(state: AssistantState) -> AssistantState:
    """
    Nodo de búsqueda web: Obtiene información actualizada de internet.
    Single Responsibility: Recuperación de información desde fuente externa.
    """
    # Solo ejecutar si está en la lista de agentes activos
    if AgentType.WEB_SEARCH.value not in state["active_agents"]:
        return state
    
    trace_id = state["conversation_id"]
    start_time = time.time()
    logger.info(f"[{trace_id}] Ejecutando agente de búsqueda web")
    
    # Incrementar contador de pasos
    state["current_step"] += 1
    
    # Verificar límites
    if state["current_step"] > state["max_steps"]:
        logger.warning(f"[{trace_id}] Se alcanzó el límite máximo de pasos ({state['max_steps']})")
        state["warnings"].append("Flujo truncado por exceder máximo de pasos")
        return state
    
    # Razonamiento ReAct
    thought = f"La consulta '{state['query']}' puede requerir información actualizada. Realizaré una búsqueda web específica."
    action = "Buscar en internet usando Tavily API con filtros para turismo en Curazao"
    
    # Registrar paso ReAct
    react_step = {
        "agent": AgentType.WEB_SEARCH.value,
        "thought": thought,
        "action": action,
        "observation": "Realizando búsqueda web...",
        "timestamp": time.time()
    }
    state["react_steps"].append(react_step)
    
    try:
        # Realizar búsqueda web con servicio especializado
        search_results = await web_service.search(state["query"], trace_id)
        processing_time = time.time() - start_time
        
        # Guardar resultados y actualizar estadísticas
        state["search_results"] = search_results
        state["processing_stats"][AgentType.WEB_SEARCH.value] = {
            "time": processing_time,
            "tokens": search_results.get("tokens", {"prompt": 0, "completion": 0, "reasoning": 0}),
            "success": not search_results.get("fallback", False) and "error" not in search_results
        }
        
        if "error" in search_results:
            state["processing_stats"][AgentType.WEB_SEARCH.value]["error"] = search_results["error"]
            state["errors"].append(f"Error en búsqueda web: {search_results['error']}")
        
        # Observación para ReAct
        observation = f"Búsqueda web completada"
        if "raw" in search_results:
            observation += f" con {len(search_results.get('raw', []))} resultados"
        
        if search_results.get("fallback", False):
            observation += " (usando sistema de respaldo)"
        
        state["react_steps"][-1]["observation"] = observation
        
        logger.info(f"[{trace_id}] Búsqueda web completada en {processing_time:.2f}s")
        
        # Si es consulta de información, generar respuesta aquí
        if state["query_type"] == "informacion":
            await generate_info_response(state)
        
    except Exception as e:
        # Manejo de errores
        processing_time = time.time() - start_time
        error_msg = f"Error en agente de búsqueda web: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}")
        
        state["errors"].append(error_msg)
        state["react_steps"][-1]["observation"] = f"Error: {str(e)}. No se pudo obtener información web."
        state["processing_stats"][AgentType.WEB_SEARCH.value] = {
            "time": processing_time,
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "success": False,
            "error": str(e)
        }
        
        # Fallback: respuesta vacía pero con estructura
        state["search_results"] = {
            "formatted": "No se pudo obtener información web en este momento.",
            "raw": [],
            "fallback": True,
            "error": str(e)
        }
    
    return state

async def generate_info_response(state: AssistantState) -> None:
    """Función auxiliar para generar respuesta informativa con fuentes combinadas"""
    trace_id = state["conversation_id"]
    start_time = time.time()
    
    # Combinar información de RAG (destacando fuente para debug)
    rag_info = ""
    for chunk in state.get("rag_results", {}).get("chunks", []):
        source = chunk.get('origin', 'unknown')
        rag_info += f"--- {chunk.get('title', 'Base de conocimiento')} [{source}] ---\n"
        rag_info += f"{chunk.get('content', '')}\n\n"
    
    web_info = state.get("search_results", {}).get("formatted", "")
    
    # Incluir estadísticas para logging
    stats = state.get("rag_results", {}).get("stats", {})
    stats_info = ""
    if stats:
        stats_info = f"""
        Fuentes: 
        - pgvector: {stats.get('pgvector_results', 0)} resultados en {stats.get('pgvector_time', 0):.3f}s
        - FAISS: {stats.get('faiss_results', 0)} resultados en {stats.get('faiss_time', 0):.3f}s
        """
        logger.info(f"[{trace_id}] Estadísticas RAG: {stats}")
    
    # Prompt para generar respuesta combinada
    combined_prompt = f"""
    El usuario ha preguntado sobre Curazao: "{state['query']}"
    
    Utiliza la siguiente información para generar una respuesta completa:
    
    INFORMACIÓN DE BASE DE CONOCIMIENTO LOCAL:
    {rag_info[:2000]}
    
    INFORMACIÓN DE BÚSQUEDA WEB RECIENTE:
    {web_info[:2000]}
    
    {stats_info}
    
    Instrucciones:
    1. Integra ambas fuentes de información de manera coherente.
    2. Prioriza la información web por ser más actualizada cuando haya conflictos.
    3. Sé conciso pero informativo.
    4. Responde directamente a la consulta del usuario.
    5. Si falta información relevante, indícalo honestamente.
    6. Incluye datos específicos relevantes (nombres de lugares, precios, horarios) cuando estén disponibles.
    
    Si ambas fuentes tienen poca información relevante a la consulta específica, 
    indícalo y proporciona la mejor respuesta posible con lo disponible.
    """
    
    try:
        response = await llm_service.chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente turístico especializado en Curazao que proporciona información precisa y útil."},
                {"role": "user", "content": combined_prompt}
            ]
        )

        # Guardar respuesta generada
        state["final_response"] = response["content"]

        # Registrar estadísticas
        processing_time = time.time() - start_time
        state["processing_stats"]["response_generation"] = {
            "time": processing_time,
            "tokens": {
                "prompt": response["usage"]["prompt_tokens"],
                "completion": response["usage"]["completion_tokens"],
                "reasoning": response["usage"]["prompt_tokens"]
            },
            "success": True,
            "provider": response["provider"]
        }
        
        logger.info(f"[{trace_id}] Respuesta informativa generada en {processing_time:.2f}s")
        
    except Exception as e:
        error_msg = f"Error generando respuesta informativa: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}")
        state["errors"].append(error_msg)
        
        # Fallback simple
        state["final_response"] = f"Lo siento, no pude procesar la información sobre tu consulta de Curazao. Por favor, intenta reformular tu pregunta."

async def itinerary_agent(state: AssistantState) -> AssistantState:
    """
    Nodo de itinerario: Genera planes de viaje personalizados.
    Single Responsibility: Planificación de itinerarios.
    """
    # Solo ejecutar si está en la lista de agentes activos
    if AgentType.ITINERARY.value not in state["active_agents"]:
        return state
    
    trace_id = state["conversation_id"]
    start_time = time.time()
    logger.info(f"[{trace_id}] Ejecutando agente de itinerario")
    
    # Incrementar contador de pasos
    state["current_step"] += 1
    
    # Verificar límites
    if state["current_step"] > state["max_steps"]:
        logger.warning(f"[{trace_id}] Se alcanzó el límite máximo de pasos ({state['max_steps']})")
        state["warnings"].append("Flujo truncado por exceder máximo de pasos")
        return state
    
    # PASO 1: Extraer preferencias del usuario (ReAct: Reasoning)
    thought = "Para crear un itinerario personalizado, primero debo identificar las preferencias específicas del usuario."
    action = "Analizar la consulta para extraer parámetros de viaje"
    
    # Registrar paso ReAct
    react_step = {
        "agent": AgentType.ITINERARY.value,
        "thought": thought,
        "action": action,
        "observation": "Analizando preferencias...",
        "timestamp": time.time()
    }
    state["react_steps"].append(react_step)
    
    try:
        # Prompt para extraer preferencias estructuradas
        preferences_prompt = f"""
        Analiza la siguiente consulta sobre un viaje a Curazao:
        
        "{state['query']}"
        
        Extrae las preferencias del usuario para crear un itinerario personalizado.
        Incluso si la información está implícita o incompleta, haz tu mejor estimación.
        
        Devuelve el resultado en formato JSON con estos campos (todos opcionales):
        - duracion: número estimado de días para el viaje
        - presupuesto: nivel de presupuesto (bajo, medio, alto) o monto aproximado
        - intereses: lista de intereses específicos mencionados (playas, historia, gastronomía, etc.)
        - tipo_viajero: composición del grupo (solo, pareja, familia con niños, amigos, etc.)
        - temporada: época del año para el viaje si se menciona 
        - alojamiento: preferencias de hospedaje si se mencionan
        
        Si alguna información no está presente, omite ese campo o déjalo como null.
        """
        
        # Estructurar salida como JSON
        preferences_response = await llm_service.chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un sistema especializado en análisis de consultas de viaje. Extraes parámetros con precisión."},
                {"role": "user", "content": preferences_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        # Parsear respuesta
        preferences = json.loads(preferences_response["content"])
        
        # Validar y normalizar estructura
        if not isinstance(preferences, dict):
            preferences = {}
        
        # Asegurar que intereses sea siempre una lista
        if "intereses" not in preferences or not isinstance(preferences["intereses"], list):
            preferences["intereses"] = []
        
        # Guardar en estado
        state["user_preferences"] = preferences
        
        # Actualizar observación ReAct
        interests_str = ", ".join(preferences.get("intereses", [])[:3]) if preferences.get("intereses") else "no especificados"
        duration_str = preferences.get("duracion", "no especificada")
        
        observation = f"Preferencias identificadas: duración={duration_str}, intereses principales={interests_str}"
        if preferences.get("tipo_viajero"):
            observation += f", viaje en {preferences.get('tipo_viajero')}"
        
        state["react_steps"][-1]["observation"] = observation
        
        # PASO 2: Generar itinerario
        itinerary_thought = "Con las preferencias identificadas, ahora crearé un itinerario personalizado combinando toda la información disponible."
        itinerary_action = "Generar plan de viaje detallado"
        
        # Registrar nuevo paso ReAct
        itinerary_step = {
            "agent": AgentType.ITINERARY.value,
            "thought": itinerary_thought,
            "action": itinerary_action,
            "observation": "Creando itinerario...",
            "timestamp": time.time()
        }
        state["react_steps"].append(itinerary_step)
        
        # Combinar información de RAG y web
        rag_chunks = []
        for chunk in state.get("rag_results", {}).get("chunks", []):
            content = chunk.get("content", "")
            if content:
                category = chunk.get("category", "general")
                rag_chunks.append(f"[{category.upper()}] {content}")
        
        rag_info = "\n\n".join(rag_chunks[:5])  # Limitar cantidad para controlar tokens
        web_info = state.get("search_results", {}).get("formatted", "")[:1500]  # Limitar tamaño
        
        # Crear prompt para itinerario
        itinerary_prompt = f"""
        Crea un itinerario personalizado para un viaje a Curazao basado en:
        
        1. CONSULTA DEL USUARIO:
        "{state['query']}"
        
        2. PREFERENCIAS IDENTIFICADAS:
        {json.dumps(preferences, indent=2)}
        
        3. INFORMACIÓN LOCAL:
        {rag_info}
        
        4. INFORMACIÓN WEB:
        {web_info}
        
        Instrucciones:
        - Si la duración no está especificada, sugiere un itinerario de 4-5 días.
        - Organiza el itinerario día por día con actividades, recomendaciones de comida y alojamiento.
        - Incluye consejos prácticos personalizados según las preferencias.
        - Sé específico con nombres de lugares, restaurantes y experiencias recomendadas.
        - Adapta las actividades al tipo de viajero.
        - Incluye una estimación de presupuesto si es relevante.
        - Prioriza las experiencias que coincidan con los intereses identificados.
        - Considera el clima según la temporada mencionada (si aplica).
        - Sugiere opciones de transporte entre lugares.
        
        Formato:
        - Comienza con una breve introducción personalizada.
        - Luego estructura el itinerario día por día.
        - Termina con consejos prácticos y recomendaciones finales.
        """
        
        # Generar itinerario
        itinerary_response = await llm_service.chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto planificador de viajes especializado en Curazao con amplio conocimiento local."},
                {"role": "user", "content": itinerary_prompt}
            ],
            temperature=0.7
        )

        # Guardar itinerario
        itinerary_content = itinerary_response["content"]
        
        state["itinerary_plan"] = {
            "plan": itinerary_content,
            "preferences": preferences,
            "timestamp": datetime.now().isoformat()
        }
        
        # Definir la respuesta final directamente
        state["final_response"] = itinerary_content
        
        # Actualizar observación final
        state["react_steps"][-1]["observation"] = "Itinerario personalizado creado exitosamente"
        
        # Calcular estadísticas
        processing_time = time.time() - start_time
        pref_prompt_tokens = preferences_response["usage"]["prompt_tokens"]
        pref_completion_tokens = preferences_response["usage"]["completion_tokens"]
        itin_prompt_tokens = itinerary_response["usage"]["prompt_tokens"]
        itin_completion_tokens = itinerary_response["usage"]["completion_tokens"]

        # Guardar estadísticas
        state["processing_stats"][AgentType.ITINERARY.value] = {
            "time": processing_time,
            "tokens": {
                "prompt": pref_prompt_tokens + itin_prompt_tokens,
                "completion": pref_completion_tokens + itin_completion_tokens,
                "reasoning": pref_prompt_tokens + itin_prompt_tokens
            },
            "success": True,
            "provider": itinerary_response["provider"]
        }
        
        logger.info(f"[{trace_id}] Itinerario generado en {processing_time:.2f}s")

        # ========================================
        # ACCIÓN AUTOMATIZADA: Email + Webhook
        # ========================================

        # Obtener email del usuario si está disponible
        user_email = state.get("user_email")

        # Enviar email con itinerario
        if email_service and user_email:
            try:
                email_sent = await email_service.send_itinerary(
                    to_email=user_email,
                    itinerary=itinerary_content,
                    query=state["query"],
                    conversation_id=trace_id
                )
                if email_sent:
                    logger.info(f" [{trace_id}] Email enviado a {user_email}")
                    state.setdefault("actions_taken", []).append("email_sent")
            except Exception as email_error:
                logger.error(f" [{trace_id}] Error enviando email: {email_error}")
                state["warnings"].append(f"No se pudo enviar email: {email_error}")

        # Disparar webhook de itinerario generado
        if webhook_service:
            try:
                webhook_sent = await webhook_service.trigger_itinerary_generated(
                    conversation_id=trace_id,
                    query=state["query"],
                    itinerary=itinerary_content,
                    user_email=user_email
                )
                if webhook_sent:
                    logger.info(f" [{trace_id}] Webhook disparado para itinerario generado")
                    state.setdefault("actions_taken", []).append("webhook_sent")
            except Exception as webhook_error:
                logger.error(f" [{trace_id}] Error enviando webhook: {webhook_error}")

    except Exception as e:
        # Manejo de errores
        processing_time = time.time() - start_time
        error_msg = f"Error en agente de itinerario: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}")
        
        state["errors"].append(error_msg)
        state["react_steps"][-1]["observation"] = f"Error: {str(e)}. No se pudo generar el itinerario."
        
        # Estadísticas en caso de error
        state["processing_stats"][AgentType.ITINERARY.value] = {
            "time": processing_time,
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "success": False,
            "error": str(e)
        }
        
        # Crear respuesta fallback simple
        state["final_response"] = "Lo siento, no pude crear un itinerario completo en este momento. Por favor, intenta especificar más detalles como duración del viaje, intereses principales y tipo de viajero para que pueda ayudarte mejor."
    
    return state

async def conversational_agent(state: AssistantState) -> AssistantState:
    """
    Nodo conversacional: Genera respuestas amigables y formatea la salida final.
    Single Responsibility: Interacción conversacional con el usuario.
    """
    trace_id = state["conversation_id"]
    start_time = time.time()
    logger.info(f"[{trace_id}] Ejecutando agente conversacional")
    
    # Incrementar contador de pasos
    state["current_step"] += 1
    
    # Verificar si ya hay una respuesta final y no es una consulta conversacional
    if state["query_type"] != "conversacional" and "final_response" in state and state["final_response"]:
        # Solo ajustar formato si es necesario, no generar nueva respuesta
        # Registrar estadísticas
        state["processing_stats"][AgentType.CONVERSATIONAL.value] = {
            "time": 0.01,  # Tiempo simbólico
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "success": True
        }
        return state
    
    # Para consultas conversacionales, generar respuesta amigable
    thought = "Generando respuesta conversacional apropiada y personalizada al contexto"
    action = "Formular respuesta natural y amigable"
    
    # Registrar paso ReAct
    react_step = {
        "agent": AgentType.CONVERSATIONAL.value,
        "thought": thought,
        "action": action,
        "observation": "Formulando respuesta...",
        "timestamp": time.time()
    }
    state["react_steps"].append(react_step)
    
    try:
        # Preparar contexto de conversación anterior limitado
        conversation_context = ""
        if state["conversation_history"]:
            # Limitar a últimas 3 interacciones para controlar tokens
            recent_history = state["conversation_history"][-3:]
            for i, exchange in enumerate(recent_history):
                conversation_context += f"Usuario: {exchange.get('user', '')}\n"
                if "assistant" in exchange:
                    conversation_context += f"Asistente: {exchange.get('assistant', '')}\n"
            
            conversation_context = f"Contexto de conversación reciente:\n{conversation_context}\n"
        
        # Prompt para respuesta conversacional
        conv_prompt = f"""
        {conversation_context}
        El usuario envió este mensaje: "{state['query']}"
        
        Genera una respuesta conversacional apropiada como asistente turístico especializado en Curazao.
        
        Instrucciones:
        - Mantén un tono amigable y servicial.
        - Sé breve pero informativo.
        - Si es un saludo o consulta simple, responde de manera natural.
        - Menciona que puedes ayudar con información sobre Curazao y planificación de viajes.
        - Si has detectado algún error técnico, discúlpate brevemente sin entrar en detalles técnicos.
        """
        
        # Agregar información sobre errores/warnings si existen
        if state["errors"] or state["warnings"]:
            conv_prompt += "\n\nNota: Se han detectado algunos problemas técnicos. Ofrece brevemente disculpas sin mencionar detalles específicos."
        
        # Generar respuesta
        response = await llm_service.chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente turístico amigable especializado en Curazao. Tu objetivo es ser útil, agradable y natural en la conversación."},
                {"role": "user", "content": conv_prompt}
            ],
            temperature=0.7
        )

        # Guardar respuesta
        state["final_response"] = response["content"]
        
        # Actualizar historial de conversación (MA-10)
        state["conversation_history"].append({
            "user": state["query"],
            "assistant": state["final_response"],
            "timestamp": time.time()
        })
        
        # Limitar tamaño del historial para controlar memoria
        if len(state["conversation_history"]) > 10:
            state["conversation_history"] = state["conversation_history"][-10:]
        
        # Actualizar observación ReAct
        state["react_steps"][-1]["observation"] = "Respuesta conversacional generada exitosamente"
        
        # Registrar estadísticas
        processing_time = time.time() - start_time
        state["processing_stats"][AgentType.CONVERSATIONAL.value] = {
            "time": processing_time,
            "tokens": {
                "prompt": response["usage"]["prompt_tokens"],
                "completion": response["usage"]["completion_tokens"],
                "reasoning": response["usage"]["prompt_tokens"]
            },
            "success": True,
            "provider": response["provider"]
        }
        
        # Calcular estadísticas totales
        total_time = sum(agent["time"] for agent_name, agent in state["processing_stats"].items() if "time" in agent)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        
        for agent_name, stats in state["processing_stats"].items():
            if "tokens" in stats:
                total_prompt_tokens += stats["tokens"].get("prompt", 0)
                total_completion_tokens += stats["tokens"].get("completion", 0) 
                total_reasoning_tokens += stats["tokens"].get("reasoning", 0)
        
        # Guardar métricas agregadas
        state["processing_stats"]["total"] = {
            "time": total_time,
            "tokens": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "reasoning": total_reasoning_tokens,
                "total": total_prompt_tokens + total_completion_tokens
            },
            "react_steps": len(state["react_steps"])
        }
        
        logger.info(f"[{trace_id}] Respuesta conversacional generada en {processing_time:.2f}s")
        
    except Exception as e:
        # Manejo de errores
        processing_time = time.time() - start_time
        error_msg = f"Error en agente conversacional: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}")
        
        state["errors"].append(error_msg)
        state["react_steps"][-1]["observation"] = f"Error: {str(e)}. Usando respuesta de emergencia."
        
        # Respuesta de emergencia
        state["final_response"] = "¡Hola! Soy tu asistente para Curazao. Lo siento, estoy teniendo dificultades técnicas en este momento. ¿Podrías intentar reformular tu consulta o preguntar algo más sobre este hermoso destino caribeño?"
        
        # Estadísticas en caso de error
        state["processing_stats"][AgentType.CONVERSATIONAL.value] = {
            "time": processing_time,
            "tokens": {"prompt": 0, "completion": 0, "reasoning": 0},
            "success": False,
            "error": str(e)
        }
    
    return state

# Paralelizar RAG y búsqueda web (mejora de rendimiento)
async def parallel_info_search(state: AssistantState) -> AssistantState:
    """Ejecuta RAG y búsqueda web en paralelo para reducir latencia"""
    trace_id = state["conversation_id"]
    logger.info(f"[{trace_id}] Ejecutando búsqueda de información en paralelo")
    
    # Crear las tareas para ejecutar en paralelo
    rag_task = asyncio.create_task(rag_agent(state.copy()))  # Usar copia para evitar conflictos de escritura
    web_task = asyncio.create_task(web_search_agent(state.copy()))
    
    # Esperar a que ambas tareas terminen
    rag_state, web_state = await asyncio.gather(rag_task, web_task)
    
    # Fusionar resultados en el estado principal
    state["rag_results"] = rag_state["rag_results"]
    state["search_results"] = web_state["search_results"]
    state["processing_stats"][AgentType.RAG.value] = rag_state["processing_stats"][AgentType.RAG.value]
    state["processing_stats"][AgentType.WEB_SEARCH.value] = web_state["processing_stats"][AgentType.WEB_SEARCH.value]
    
    # Combinar errores y advertencias
    if "errors" in rag_state:
        state["errors"].extend(rag_state["errors"])
    if "errors" in web_state:
        state["errors"].extend(web_state["errors"])
    if "warnings" in rag_state:
        state["warnings"].extend(rag_state["warnings"])
    if "warnings" in web_state:
        state["warnings"].extend(web_state["warnings"])
    
    # Tomar la respuesta de la búsqueda web si ya generó una
    if "final_response" in web_state and web_state["final_response"]:
        state["final_response"] = web_state["final_response"]
        if "response_generation" in web_state["processing_stats"]:
            state["processing_stats"]["response_generation"] = web_state["processing_stats"]["response_generation"]
    
    # Unir pasos ReAct
    all_steps = state["react_steps"]
    for step in rag_state["react_steps"]:
        if step not in all_steps:
            all_steps.append(step)
    for step in web_state["react_steps"]:
        if step not in all_steps:
            all_steps.append(step)
    
    # Ordenar pasos por timestamp
    state["react_steps"] = sorted(all_steps, key=lambda x: x["timestamp"])
    
    # Actualizar contador de pasos
    state["current_step"] = max(rag_state["current_step"], web_state["current_step"])
    
    return state

# Router para toma de decisiones especializado (LG-3)
async def agent_router(state: AssistantState) -> str:
    """
    Router que determina el siguiente nodo basado en el tipo de consulta y agentes activos.
    Implementa lógica de enrutamiento estructurada con paralelización.
    """
    query_type = state["query_type"]
    
    # Verificar límites de pasos
    if state["current_step"] >= state["max_steps"]:
        logger.warning(f"[{state['conversation_id']}] Limite de pasos alcanzado: {state['current_step']}/{state['max_steps']}")
        return AgentType.CONVERSATIONAL.value
    
    # Lógica de enrutamiento mejorado con paralelización
    if query_type == "conversacional":
        # Flujo simple para consultas conversacionales
        return AgentType.CONVERSATIONAL.value
    elif query_type == "informacion":
        # Paralelizar RAG y búsqueda web para consultas informativas
        if state["current_step"] == 1:
            # Primero ejecutar búsqueda paralela
            await parallel_info_search(state)
            # Luego ir al agente conversacional para respuesta final
            return AgentType.CONVERSATIONAL.value
        else:
            return AgentType.CONVERSATIONAL.value
    elif query_type == "itinerario":
        # Para itinerarios, paralelizar primero la búsqueda de información
        if state["current_step"] == 1:
            await parallel_info_search(state)
            return AgentType.ITINERARY.value
        elif state["current_step"] > 1 and AgentType.ITINERARY.value in state["active_agents"]:
            return AgentType.ITINERARY.value
        else:
            return AgentType.CONVERSATIONAL.value
    else:
        # Por defecto ir al agente conversacional para respuesta final
        return AgentType.CONVERSATIONAL.value

def build_assistant_graph() -> StateGraph:
    """
    Construye el grafo del asistente con mejores prácticas de LangGraph.
    - Tipado explícito
    - Nodos con responsabilidad única
    - Router estructurado
    - Manejo de errores y límites
    """
    # Crear grafo con estado tipado (LG-2)
    graph = StateGraph(AssistantState)
    
    # Añadir nodos con responsabilidad única (LG-1)
    graph.add_node(AgentType.CLASSIFIER.value, query_classifier)
    graph.add_node(AgentType.RAG.value, rag_agent)
    graph.add_node(AgentType.WEB_SEARCH.value, web_search_agent)
    graph.add_node(AgentType.ITINERARY.value, itinerary_agent)
    graph.add_node(AgentType.CONVERSATIONAL.value, conversational_agent)
    
    # Usar router estructurado para conectar nodos (LG-3)
    graph.add_conditional_edges(
        AgentType.CLASSIFIER.value,
        agent_router,
        {
            AgentType.RAG.value: AgentType.RAG.value,
            AgentType.CONVERSATIONAL.value: AgentType.CONVERSATIONAL.value,
            AgentType.ITINERARY.value: AgentType.ITINERARY.value
        }
    )
    
    # El router ahora maneja la paralelización, pero mantener conexiones explícitas
    graph.add_conditional_edges(
        AgentType.RAG.value,
        agent_router,
        {
            AgentType.WEB_SEARCH.value: AgentType.WEB_SEARCH.value,
            AgentType.CONVERSATIONAL.value: AgentType.CONVERSATIONAL.value
        }
    )
    
    # Conectar nodo de búsqueda web
    graph.add_conditional_edges(
        AgentType.WEB_SEARCH.value,
        agent_router,
        {
            AgentType.ITINERARY.value: AgentType.ITINERARY.value,
            AgentType.CONVERSATIONAL.value: AgentType.CONVERSATIONAL.value
        }
    )
    
    # El agente de itinerario siempre va al conversacional
    graph.add_edge(AgentType.ITINERARY.value, AgentType.CONVERSATIONAL.value)
    
    # El agente conversacional siempre va al final
    graph.add_edge(AgentType.CONVERSATIONAL.value, END)
    
    # Configurar el nodo de entrada
    graph.set_entry_point(AgentType.CLASSIFIER.value)
    
    # Compilar y validar el grafo (LG-9)
    return graph.compile()

# Usar nuestra implementación personalizada en vez de la oficial
checkpoint_manager = SimpleCheckpointManager("./checkpoints")

# Función principal para procesar consultas
async def process_query(query: str, conversation_id: str = None, user_email: str = None) -> Dict[str, Any]:
    """
    Procesa una consulta a través del sistema de agentes con observabilidad
    y manejo de errores mejorado.
    """
    # Generar ID de conversación único si no se proporciona
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    trace_id = conversation_id
    logger.info(f"[{trace_id}] Procesando nueva consulta: '{query[:50]}...'")
    
    try:
        # Inicializar servicios
        await vector_service.initialize()
        
        # Crear grafo si no existe 
        graph = build_assistant_graph()
        
        # Estado inicial con información completa
        initial_state = {
            # Datos de entrada
            "query": query,
            "conversation_id": conversation_id,
            "timestamp": time.time(),
            "user_email": user_email,  # Para acciones automatizadas

            # Estado de ejecución
            "active_agents": set(),
            "query_type": "conversacional",  # Valor por defecto
            "current_step": 0,
            "max_steps": MAX_STEPS,

            # Resultados de agentes
            "rag_results": {},
            "search_results": {},
            "itinerary_plan": {},
            "user_preferences": {},

            # Historial y memoria
            "conversation_history": [],  # Se recuperará del checkpoint si existe
            "react_steps": [],

            # Salida y métricas
            "final_response": "",
            "processing_stats": {},
            "errors": [],
            "warnings": []
        }
        
        # Restaurar historial de conversación si existe (LG-6)
        try:
            checkpoint = checkpoint_manager.get(conversation_id)
            if checkpoint and "conversation_history" in checkpoint:
                initial_state["conversation_history"] = checkpoint["conversation_history"]
                logger.info(f"[{trace_id}] Historial de conversación restaurado: {len(checkpoint['conversation_history'])} mensajes")
        except Exception as e:
            logger.warning(f"[{trace_id}] Error restaurando checkpoint: {e}")
        
        # Ejecutar grafo con timeout de seguridad (MA-8)
        start_time = time.time()
        
        try:
            # Ejecutar con timeout para evitar bloqueos
            result = await asyncio.wait_for(
                graph.ainvoke(initial_state),
                timeout=TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"[{trace_id}] Timeout procesando consulta después de {TIMEOUT_SECONDS}s")
            # Crear estado de resultado parcial
            result = initial_state.copy()
            result["errors"].append(f"Timeout después de {TIMEOUT_SECONDS}s")
            result["final_response"] = "Lo siento, estoy tardando demasiado en procesar tu consulta. ¿Podrías intentar con una pregunta más específica sobre Curazao?"
            
            # Estadísticas básicas en caso de timeout
            result["processing_stats"]["total"] = {
                "time": TIMEOUT_SECONDS,
                "tokens": {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0},
                "react_steps": 0,
                "timeout": True
            }
        
        total_time = time.time() - start_time
        
        # Guardar checkpoint de historia (LG-6)
        try:
            checkpoint_manager.put(
                conversation_id,
                {"conversation_history": result.get("conversation_history", [])}
            )
        except Exception as e:
            logger.warning(f"[{trace_id}] Error guardando checkpoint: {e}")
            result["warnings"].append("No se pudo guardar el historial de conversación")
        
        # Formatear respuesta para cliente
        response = {
            "response": result["final_response"],
            "stats": result["processing_stats"],
            "query_type": result["query_type"],
            "active_agents": list(result["active_agents"]),
            "total_time": total_time,
            "conversation_id": conversation_id,
            "errors": len(result["errors"]) > 0,
            "warnings": result["warnings"],
            "react_steps_count": len(result["react_steps"])
        }
        
        # Incluir desglose detallado de tokens si está disponible
        if "total" in result["processing_stats"]:
            tokens = result["processing_stats"]["total"].get("tokens", {})
            response["tokens"] = {
                "prompt": tokens.get("prompt", 0),
                "completion": tokens.get("completion", 0), 
                "reasoning": tokens.get("reasoning", 0),
                "total": tokens.get("total", 0)
            }
        
        logger.info(f"[{trace_id}] Consulta procesada exitosamente en {total_time:.2f}s")

        # ========================================
        # WEBHOOK: Consulta completada
        # ========================================
        if webhook_service:
            try:
                await webhook_service.trigger_query_completed(
                    conversation_id=conversation_id,
                    query=query,
                    query_type=result["query_type"],
                    response_preview=result["final_response"][:200],
                    processing_time=total_time,
                    tokens_used=response.get("tokens", {}).get("total", 0)
                )
                logger.info(f" [{trace_id}] Webhook de consulta completada disparado")
            except Exception as webhook_error:
                logger.warning(f"️ [{trace_id}] Error en webhook: {webhook_error}")

        return response
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error fatal procesando consulta: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Respuesta de emergencia (MA-5)
        return {
            "response": "Lo siento, ha ocurrido un error inesperado en el asistente. Por favor, intenta más tarde con otra consulta sobre Curazao.",
            "error": str(e),
            "conversation_id": conversation_id,
            "total_time": time.time() - (start_time if 'start_time' in locals() else time.time()),
            "critical_error": True
        }
