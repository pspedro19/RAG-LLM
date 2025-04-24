# app/core/config/config.py
import os
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from dotenv import load_dotenv

# Clases de configuración independientes para app
class FAISSConfig:
    """Configuración para FAISS."""
    
    def __init__(
        self,
        vector_size: int = 384,
        n_lists: int = 100,
        nprobe: int = 10,
        indices_dir: Path = None,
        backup_dir: Path = None
    ):
        self.vector_size = vector_size
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.indices_dir = indices_dir
        self.backup_dir = backup_dir

class DatabaseConfig:
    """Configuración para la base de datos."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        username: str = "myuser",
        password: str = "mypassword",
        database: str = "mydatabase"
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        
    def get_connection_string(self) -> str:
        """Retorna el string de conexión para la base de datos."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class ModelConfig:
    """Configuración para modelos de embeddings."""
    
    def __init__(self, name: str, vector_size: int, index_path: str, n_lists: int = 100):
        self.name = name
        self.vector_size = vector_size
        self.index_path = index_path
        self.n_lists = n_lists

class Config:
    """Configuración unificada para el sistema RAG con FAISS y pgvector."""
    
    def __init__(self, **kwargs: Any):
        """Inicializa y carga configuración, permitiendo override para tests."""
        # Definir directorios base
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.PROJECT_ROOT = self.BASE_DIR.parent
        self.INDICES_DIR = self.BASE_DIR / "data/indices"
        self.KB_DIR = self.BASE_DIR / "data/documents"
        self.BACKUP_DIR = self.INDICES_DIR / "backups"
        
        # Asegurar que los directorios existan
        os.makedirs(self.INDICES_DIR, exist_ok=True)
        os.makedirs(self.KB_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
        
        # Configuración de chunking
        self.CHUNK_SIZE = 1500
        self.CHUNK_OVERLAP = 300
        
        # Database (valores por defecto)
        self.DB_HOST = "postgres_pgvector"
        self.DB_PORT = 5432
        self.DB_NAME = "mydatabase"
        self.DB_USER = "myuser"
        self.DB_PASSWORD = "mypassword"
        
        # Configuración de FAISS
        self.VECTOR_SIZE = 384  # Dimensión por defecto (miniLM)
        self.N_LISTS = 100      # Número de listas para IVF si se usa
        
        # Configuración de sincronización
        self.SYNC_BATCH_SIZE = 1000
        self.AUTO_SYNC = True
        self.SYNC_INTERVAL = 300
        
        # Configuración de búsqueda
        self.DEFAULT_TOP_K = 5
        self.NPROBE = 10  # Parámetro para índices IVF
        
        # Configuración de búsqueda cruzada bilingüe
        self.ENABLE_CROSS_LINGUAL = True  # Habilita búsqueda en ambos idiomas
        self.ENABLE_QUERY_EXPANSION = False  # Opcional: expansión de consulta
        
        # Intentar cargar .env si existe
        env_path = self.PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Sobrescribir configuraciones con variables de entorno
        self._load_from_env()
        
        # Sobrescribir configuraciones con kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Inicializar modelos después de cargar configuraciones
        self.MODELS = {
            'miniLM': ModelConfig(
                name="all-MiniLM-L6-v2",
                vector_size=384,
                index_path="minilm_index.faiss",
            ),
            'mpnet': ModelConfig(
                name="multi-qa-mpnet-base-dot-v1",
                vector_size=768,
                index_path="mpnet_index.faiss",
            )
        }
        
        # Actualizar rutas de índices con directorio base
        for model_key, model_config in self.MODELS.items():
            model_config.index_path = str(self.INDICES_DIR / model_config.index_path)
    
    def _load_from_env(self):
        """Carga valores desde variables de entorno."""
        env_mappings = {
            "DB_HOST": ("DB_HOST", str),
            "DB_PORT": ("DB_PORT", int),
            "DB_NAME": ("DB_NAME", str),
            "DB_USER": ("DB_USER", str),
            "DB_PASSWORD": ("DB_PASSWORD", str),
            "CHUNK_SIZE": ("CHUNK_SIZE", int),
            "CHUNK_OVERLAP": ("CHUNK_OVERLAP", int),
            "SYNC_BATCH_SIZE": ("SYNC_BATCH_SIZE", int),
            "AUTO_SYNC": ("AUTO_SYNC", lambda v: v.lower() in ('true', '1', 't', 'yes', 'y') if isinstance(v, str) else bool(v)),
            "SYNC_INTERVAL": ("SYNC_INTERVAL", int),
            "DEFAULT_TOP_K": ("DEFAULT_TOP_K", int),
            "NPROBE": ("NPROBE", int),
            "ENABLE_CROSS_LINGUAL": ("ENABLE_CROSS_LINGUAL", lambda v: v.lower() in ('true', '1', 't', 'yes', 'y') if isinstance(v, str) else bool(v)),
            "ENABLE_QUERY_EXPANSION": ("ENABLE_QUERY_EXPANSION", lambda v: v.lower() in ('true', '1', 't', 'yes', 'y') if isinstance(v, str) else bool(v)),
        }
        
        for attr, (env_var, type_func) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                try:
                    setattr(self, attr, type_func(value))
                except (ValueError, TypeError) as e:
                    print(f"Error al convertir {env_var}={value} usando {type_func}: {e}")
    
    def get_model_config(self, model_key: str = 'miniLM') -> ModelConfig:
        """Obtiene la configuración para un modelo específico."""
        return self.MODELS.get(model_key, self.MODELS['miniLM'])
    
    def default_model_config(self) -> ModelConfig:
        """Retorna la configuración del modelo por defecto (miniLM)."""
        return self.MODELS['miniLM']
    
    def get_db_config(self) -> DatabaseConfig:
        """Obtiene la configuración de base de datos."""
        return DatabaseConfig(
            host=self.DB_HOST,
            port=self.DB_PORT,
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            database=self.DB_NAME
        )
    
    def get_faiss_config(self) -> FAISSConfig:
        """Obtiene la configuración de FAISS."""
        return FAISSConfig(
            vector_size=self.VECTOR_SIZE,
            n_lists=self.N_LISTS,
            nprobe=self.NPROBE,
            indices_dir=self.INDICES_DIR,
            backup_dir=self.BACKUP_DIR
        )

# Instancia singleton para acceso global
_config = None

def get_config() -> Config:
    """Obtiene o crea una instancia singleton de la configuración."""
    global _config
    
    if _config is None:
        _config = Config()
    
    return _config