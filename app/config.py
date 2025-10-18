"""
Configuration Module - Centralized configuration management
Carga todas las configuraciones desde variables de entorno
"""

import os
import sys
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ==========================================
# API Keys Configuration
# ==========================================
class APIConfig:
    """Configuración de APIs externas"""

    # OpenAI (OBLIGATORIO)
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

    # HuggingFace (OBLIGATORIO para embeddings)
    HUGGINGFACE_TOKEN: str = os.environ.get("HUGGINGFACE_TOKEN", "")
    HUGGINGFACE_EMBEDDING_MODEL: str = os.environ.get(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )

    # Claude/Anthropic (OPCIONAL)
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Tavily Search (OPCIONAL)
    TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

    # SendGrid (OPCIONAL)
    SENDGRID_API_KEY: str = os.environ.get("SENDGRID_API_KEY", "")

    @classmethod
    def validate(cls) -> bool:
        """Valida que las APIs críticas estén configuradas"""
        # Verificar que al menos uno de los proveedores LLM esté configurado
        has_llm_provider = bool(cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY)

        if not has_llm_provider:
            logger.error("⚠️ CRÍTICO: No hay proveedores de LLM configurados")
            logger.error("Configura al menos una de las siguientes variables:")
            logger.error("  - OPENAI_API_KEY=sk-...")
            logger.error("  - ANTHROPIC_API_KEY=sk-ant-...")
            return False

        # Log de proveedores disponibles
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("OpenAI")
        if cls.ANTHROPIC_API_KEY:
            providers.append("Claude (Anthropic)")

        logger.info(f"✓ Proveedores LLM disponibles: {', '.join(providers)}")

        if not cls.HUGGINGFACE_TOKEN:
            logger.warning("HUGGINGFACE_TOKEN no configurado. Algunos modelos pueden no descargarse.")

        return True

    @classmethod
    def get_llm_provider(cls) -> str:
        """Retorna el proveedor de LLM primario a usar (openai o anthropic)"""
        # Prioridad: OpenAI primero, luego Anthropic
        if cls.OPENAI_API_KEY:
            return "openai"
        elif cls.ANTHROPIC_API_KEY:
            return "anthropic"
        return "none"

    @classmethod
    def has_fallback_provider(cls) -> bool:
        """Verifica si hay un proveedor de fallback disponible"""
        return bool(cls.OPENAI_API_KEY and cls.ANTHROPIC_API_KEY)


# ==========================================
# Database Configuration
# ==========================================
class DatabaseConfig:
    """Configuración de base de datos PostgreSQL"""

    HOST: str = os.environ.get("PG_HOST", "localhost")
    PORT: int = int(os.environ.get("PG_PORT", "5432"))
    DATABASE: str = os.environ.get("PG_DATABASE", os.environ.get("POSTGRES_DB", "mydatabase"))
    USER: str = os.environ.get("PG_USER", os.environ.get("POSTGRES_USER", "myuser"))
    PASSWORD: str = os.environ.get("PG_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "mypassword"))

    @classmethod
    def get_connection_string(cls) -> str:
        """Retorna string de conexión PostgreSQL"""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"


# ==========================================
# Model Configuration
# ==========================================
class ModelConfig:
    """Configuración de modelos"""

    # LLM Model
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4-turbo")
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

    # Vision Model (ViT)
    VISION_MODEL: str = os.environ.get("VISION_MODEL", "vit_b_16")
    VISION_DEVICE: str = os.environ.get("VISION_DEVICE", "auto")  # auto, cuda, cpu

    # HuggingFace Vision Model (opcional - si se especifica, se usa en lugar de torchvision)
    HUGGINGFACE_VISION_MODEL: str = os.environ.get("HUGGINGFACE_VISION_MODEL", "")
    # Ejemplos: "google/vit-base-patch16-224", "microsoft/resnet-50", "facebook/deit-base-patch16-224"

    # Embedding Model (from HuggingFace)
    EMBEDDING_MODEL: str = APIConfig.HUGGINGFACE_EMBEDDING_MODEL
    EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "auto")

    # Token Limits
    MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "12000"))
    MAX_COMPLETION_TOKENS: int = int(os.environ.get("MAX_COMPLETION_TOKENS", "2000"))


# ==========================================
# Application Configuration
# ==========================================
class AppConfig:
    """Configuración general de la aplicación"""

    # Server
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("FASTAPI_PORT", "8000"))

    # Timeouts and Limits
    TIMEOUT_SECONDS: int = int(os.environ.get("TIMEOUT_SECONDS", "150"))
    MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "15"))

    # RAG Configuration
    RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "8"))
    RAG_SIMILARITY_THRESHOLD: float = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.65"))

    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.environ.get("LOG_FILE", "curacao_assistant.log")

    # CORS
    CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")

    # Debug Mode
    DEBUG: bool = os.environ.get("DEBUG", "0") == "1"


# ==========================================
# Webhook & Notifications
# ==========================================
class NotificationConfig:
    """Configuración de notificaciones"""

    WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "")
    WEBHOOK_ENABLED: bool = bool(WEBHOOK_URL)

    EMAIL_FROM: str = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@curacao-assistant.com")
    EMAIL_ENABLED: bool = bool(APIConfig.SENDGRID_API_KEY)


# ==========================================
# Global Configuration Object
# ==========================================
class Config:
    """Clase principal de configuración"""

    api = APIConfig
    db = DatabaseConfig
    model = ModelConfig
    app = AppConfig
    notifications = NotificationConfig

    @classmethod
    def validate_all(cls) -> bool:
        """Valida toda la configuración"""
        logger.info("Validando configuración del sistema...")

        # Validar APIs
        if not cls.api.validate():
            return False

        # Log de configuración
        logger.info(f"  LLM Provider: {cls.api.get_llm_provider()}")
        logger.info(f"  LLM Model: {cls.model.LLM_MODEL}")
        logger.info(f"  Embedding Model: {cls.model.EMBEDDING_MODEL}")
        logger.info(f"  Vision Model: {cls.model.VISION_MODEL}")
        logger.info(f"  Database: {cls.db.HOST}:{cls.db.PORT}/{cls.db.DATABASE}")
        logger.info(f"  Server: {cls.app.HOST}:{cls.app.PORT}")

        return True

    @classmethod
    def get_summary(cls) -> dict:
        """Retorna resumen de configuración para health check"""
        return {
            "llm_provider": cls.api.get_llm_provider(),
            "llm_model": cls.model.LLM_MODEL,
            "embedding_model": cls.model.EMBEDDING_MODEL,
            "vision_model": cls.model.VISION_MODEL,
            "database": f"{cls.db.HOST}:{cls.db.PORT}",
            "apis_configured": {
                "openai": bool(cls.api.OPENAI_API_KEY),
                "huggingface": bool(cls.api.HUGGINGFACE_TOKEN),
                "anthropic": bool(cls.api.ANTHROPIC_API_KEY),
                "tavily": bool(cls.api.TAVILY_API_KEY),
                "sendgrid": bool(cls.api.SENDGRID_API_KEY),
            }
        }


# ==========================================
# Validation on Import
# ==========================================
def load_env():
    """Carga variables de entorno desde .env"""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Busca .env en el directorio actual
        load_dotenv(".env.minimal")  # Busca .env.minimal
        logger.info("Variables de entorno cargadas")
    except ImportError:
        logger.warning("python-dotenv no instalado. Usando solo variables del sistema.")
    except Exception as e:
        logger.warning(f"Error cargando .env: {e}")


# Cargar automáticamente al importar
load_env()

# Validar configuración
if not Config.validate_all():
    logger.error("\n" + "="*50)
    logger.error("CONFIGURACIÓN INCOMPLETA")
    logger.error("="*50)
    logger.error("Por favor configura las siguientes variables:")
    logger.error("1. OPENAI_API_KEY=sk-...")
    logger.error("2. HUGGINGFACE_TOKEN=hf_...")
    logger.error("\nCopia .env.minimal a .env y completa los valores.")
    logger.error("="*50 + "\n")
