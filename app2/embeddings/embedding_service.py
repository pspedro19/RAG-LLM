# app2/embeddings/embedding_service.py

import logging
import time
import numpy as np
from typing import List, Dict, Any, Union, Optional
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

from app2.core.config.config import Config

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """Clase base abstracta para generadores de embeddings."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos a vectorizar
            
        Returns:
            Array numpy con los embeddings generados
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensión de los vectores generados."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre identificativo del modelo."""
        pass


class LocalModelEmbedder(BaseEmbedder):
    """Generador de embeddings usando modelos locales con SentenceTransformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicializa el embedder local.
        
        Args:
            model_name: Nombre del modelo de SentenceTransformers a utilizar
        """
        self._model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Modelo {model_name} cargado correctamente")
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            # Fallback a un modelo conocido
            self._model_name = 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"Usando modelo de fallback: {self._model_name}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings usando el modelo local.
        
        Args:
            texts: Lista de textos a vectorizar
            
        Returns:
            Array numpy con los embeddings generados
        """
        # Asegurar que los textos sean strings
        texts = [str(text) for text in texts]
        
        # Generar embeddings (normalizados para similitud coseno)
        return self.model.encode(texts, normalize_embeddings=True)
    
    @property
    def dimension(self) -> int:
        """Dimensión de los vectores generados."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        """Nombre identificativo del modelo."""
        return self._model_name


class OpenAIEmbedder(BaseEmbedder):
    """Generador de embeddings usando la API de OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """
        Inicializa el embedder de OpenAI.
        
        Args:
            api_key: Clave API de OpenAI (si es None, se lee de OPENAI_API_KEY)
            model: Modelo de embeddings a utilizar
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Se requiere el paquete 'openai'. Instala con 'pip install openai'")
        
        self._model_name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Se requiere una API key de OpenAI. Proporciona como parámetro o establece OPENAI_API_KEY")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Dimensiones según modelo (podría actualizarse con más modelos)
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings usando la API de OpenAI.
        
        Args:
            texts: Lista de textos a vectorizar
            
        Returns:
            Array numpy con los embeddings generados
        """
        if not texts:
            return np.array([])
        
        # Asegurar que los textos sean strings
        texts = [str(text) for text in texts]
        
        try:
            response = self.client.embeddings.create(
                model=self._model_name,
                input=texts
            )
            
            # Extraer embeddings y organizarlos en el orden correcto
            embeddings = []
            for i in range(len(texts)):
                for emb in response.data:
                    if emb.index == i:
                        embeddings.append(emb.embedding)
                        break
            
            return np.array(embeddings, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error generando embeddings con OpenAI: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Dimensión de los vectores generados."""
        return self._dimensions.get(self._model_name, 1536)
    
    @property
    def model_name(self) -> str:
        """Nombre identificativo del modelo."""
        return self._model_name


class EmbeddingService:
    """
    Servicio central para generación de embeddings con soporte para múltiples modelos.
    Implementa caché, batching y reintentos.
    """
    
    def __init__(self, config: Config):
        """
        Inicializa el servicio de embeddings.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        
        # Configuración para batching
        self.batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 32)
        
        # Inicializar embedders según configuración
        self.embedders = {}
        
        # Cargar embedder por defecto (normalmente un modelo local)
        default_model = getattr(config, 'DEFAULT_EMBEDDING_MODEL', 'miniLM')
        self._load_default_embedders(default_model)
        
        # Caché para embeddings frecuentes
        self._setup_cache()
    
    def _load_default_embedders(self, default_model: str):
        """
        Carga los embedders predeterminados.
        
        Args:
            default_model: Nombre del modelo por defecto
        """
        # Inicializar embedders locales
        local_models = {
            'miniLM': 'all-MiniLM-L6-v2',
            'mpnet': 'multi-qa-mpnet-base-dot-v1'
        }
        
        for key, model_name in local_models.items():
            try:
                self.embedders[key] = LocalModelEmbedder(model_name)
                logger.info(f"Embedder local '{key}' inicializado con modelo {model_name}")
            except Exception as e:
                logger.warning(f"No se pudo cargar el embedder local '{key}': {e}")
        
        # Inicializar embedder OpenAI si hay API key
        if hasattr(self.config, 'OPENAI_API_KEY') and self.config.OPENAI_API_KEY:
            try:
                openai_model = getattr(self.config, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
                self.embedders['openai'] = OpenAIEmbedder(
                    api_key=self.config.OPENAI_API_KEY,
                    model=openai_model
                )
                logger.info(f"Embedder OpenAI inicializado con modelo {openai_model}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar el embedder de OpenAI: {e}")
    
    def _setup_cache(self):
        """Configura la caché para textos frecuentes."""
        cache_size = getattr(self.config, 'EMBEDDING_CACHE_SIZE', 1024)
        
        # Decorador para añadir caché
        self._get_embedding_cached = lru_cache(maxsize=cache_size)(self._get_embedding)
        logger.info(f"Caché de embeddings configurada con tamaño {cache_size}")
    
    def _get_embedding(self, text: str, model_key: str) -> np.ndarray:
        """
        Obtiene el embedding para un solo texto (función interna para caché).
        
        Args:
            text: Texto a vectorizar
            model_key: Clave del modelo a utilizar
            
        Returns:
            Array de embedding
        """
        embedder = self.get_embedder(model_key)
        return embedder.embed_texts([text])[0]
    
    def get_embedder(self, model_key: str) -> BaseEmbedder:
        """
        Obtiene un embedder específico por su clave.
        
        Args:
            model_key: Clave del modelo ('miniLM', 'openai', etc.)
            
        Returns:
            Instancia de BaseEmbedder
            
        Raises:
            ValueError: Si el modelo solicitado no está disponible
        """
        if model_key not in self.embedders:
            available = list(self.embedders.keys())
            logger.warning(f"Embedder '{model_key}' no disponible. Opciones: {available}")
            
            # Usar el primer embedder disponible como fallback
            if available:
                model_key = available[0]
                logger.info(f"Usando '{model_key}' como fallback")
            else:
                raise ValueError(f"No hay embedders disponibles")
        
        return self.embedders[model_key]
    
    def generate_embeddings(self, texts: List[str], model_key: str = None) -> np.ndarray:
        """
        Genera embeddings para una lista de textos, usando batching si es necesario.
        
        Args:
            texts: Lista de textos a vectorizar
            model_key: Clave del modelo a utilizar (None para usar el modelo por defecto)
            
        Returns:
            Array numpy con los embeddings generados
        """
        if not texts:
            return np.array([])
        
        model_key = model_key or getattr(self.config, 'DEFAULT_EMBEDDING_MODEL', 'miniLM')
        embedder = self.get_embedder(model_key)
        
        # Caso especial: un solo texto, usar caché
        if len(texts) == 1:
            try:
                return np.array([self._get_embedding_cached(texts[0], model_key)])
            except Exception as e:
                logger.warning(f"Error en caché de embeddings: {e}. Generando sin caché.")
                return embedder.embed_texts(texts)
        
        # Batching para múltiples textos
        if len(texts) <= self.batch_size:
            return embedder.embed_texts(texts)
        
        # Procesamiento por lotes para conjuntos grandes
        start_time = time.time()
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Procesando lote {i//self.batch_size + 1} ({len(batch)} textos)")
            
            try:
                batch_embeddings = embedder.embed_texts(batch)
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error procesando lote {i//self.batch_size + 1}: {e}")
                # Reintento individual si falla el lote
                for j, text in enumerate(batch):
                    try:
                        emb = embedder.embed_texts([text])[0]
                        if j == 0:
                            batch_embeddings = np.zeros((len(batch), len(emb)), dtype=np.float32)
                        batch_embeddings[j] = emb
                    except Exception as e2:
                        logger.error(f"Error con texto individual {i+j}: {e2}")
                        # Usar vector de ceros como fallback
                        if j == 0:
                            # Determinar dimensión
                            dim = embedder.dimension
                            batch_embeddings = np.zeros((len(batch), dim), dtype=np.float32)
                all_embeddings.append(batch_embeddings)
        
        # Combinar todos los lotes
        result = np.vstack(all_embeddings)
        
        elapsed = time.time() - start_time
        logger.info(f"Generados {len(texts)} embeddings en {elapsed:.2f}s ({len(texts)/elapsed:.1f} textos/s)")
        
        return result
    
    def get_models_info(self) -> List[Dict[str, Any]]:
        """
        Obtiene información sobre los modelos disponibles.
        
        Returns:
            Lista de diccionarios con información de modelos
        """
        return [
            {
                'key': key,
                'name': embedder.model_name,
                'dimension': embedder.dimension,
                'type': embedder.__class__.__name__
            }
            for key, embedder in self.embedders.items()
        ]