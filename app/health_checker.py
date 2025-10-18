"""
Health Checker Multi-Agente
Verifica el estado de todos los componentes del sistema
"""

import asyncio
import logging
from typing import Dict, List, Any
from enum import Enum
import time

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Representa el estado de salud de un componente"""

    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.message = ""
        self.checks: List[Dict[str, Any]] = []
        self.check_time = 0.0

    def add_check(self, check_name: str, passed: bool, message: str = ""):
        """Añade un check individual"""
        self.checks.append({
            "name": check_name,
            "passed": passed,
            "message": message
        })

    def compute_status(self) -> HealthStatus:
        """Calcula el estado basado en los checks"""
        if not self.checks:
            return HealthStatus.UNKNOWN

        failed = [c for c in self.checks if not c["passed"]]

        if not failed:
            return HealthStatus.HEALTHY
        elif len(failed) < len(self.checks) / 2:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        self.status = self.compute_status()
        return {
            "component": self.name,
            "status": self.status.value,
            "message": self.message,
            "checks": self.checks,
            "check_duration_ms": round(self.check_time * 1000, 2)
        }


class HealthChecker:
    """Sistema multi-agente de health checking"""

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}

    # ==========================================
    # Agent 1: Configuration Checker
    # ==========================================
    async def check_configuration(self) -> ComponentHealth:
        """Verifica que la configuración esté completa"""
        start = time.time()
        health = ComponentHealth("configuration")

        try:
            from config import Config

            # Check OpenAI API Key
            health.add_check(
                "openai_api_key",
                bool(Config.api.OPENAI_API_KEY),
                "OpenAI API key configured" if Config.api.OPENAI_API_KEY else "Missing OPENAI_API_KEY"
            )

            # Check HuggingFace Token
            health.add_check(
                "huggingface_token",
                bool(Config.api.HUGGINGFACE_TOKEN),
                "HuggingFace token configured" if Config.api.HUGGINGFACE_TOKEN else "Missing HUGGINGFACE_TOKEN (optional)"
            )

            # Check Embedding Model
            health.add_check(
                "embedding_model",
                bool(Config.model.EMBEDDING_MODEL),
                f"Using {Config.model.EMBEDDING_MODEL}"
            )

            # Check LLM Model
            health.add_check(
                "llm_model",
                bool(Config.model.LLM_MODEL),
                f"Using {Config.model.LLM_MODEL}"
            )

            health.message = f"Configuration: {Config.api.get_llm_provider()}"

        except Exception as e:
            health.add_check("config_load", False, f"Error: {str(e)}")
            health.message = "Configuration error"

        health.check_time = time.time() - start
        return health

    # ==========================================
    # Agent 2: Database Checker
    # ==========================================
    async def check_database(self) -> ComponentHealth:
        """Verifica conexión a PostgreSQL"""
        start = time.time()
        health = ComponentHealth("database")

        try:
            import psycopg2
            from config import DatabaseConfig

            # Intentar conectar
            try:
                conn = psycopg2.connect(
                    host=DatabaseConfig.HOST,
                    port=DatabaseConfig.PORT,
                    database=DatabaseConfig.DATABASE,
                    user=DatabaseConfig.USER,
                    password=DatabaseConfig.PASSWORD,
                    connect_timeout=5
                )
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                conn.close()

                health.add_check(
                    "connection",
                    result[0] == 1,
                    f"Connected to {DatabaseConfig.HOST}:{DatabaseConfig.PORT}"
                )

                # Check pgvector extension
                try:
                    conn = psycopg2.connect(
                        host=DatabaseConfig.HOST,
                        port=DatabaseConfig.PORT,
                        database=DatabaseConfig.DATABASE,
                        user=DatabaseConfig.USER,
                        password=DatabaseConfig.PASSWORD,
                        connect_timeout=5
                    )
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                    has_vector = cursor.fetchone() is not None
                    conn.close()

                    health.add_check(
                        "pgvector_extension",
                        has_vector,
                        "pgvector extension installed" if has_vector else "pgvector not installed (optional)"
                    )
                except:
                    health.add_check("pgvector_extension", False, "Could not check pgvector")

                health.message = "Database connected"

            except Exception as conn_error:
                health.add_check("connection", False, f"Connection failed: {str(conn_error)}")
                health.message = "Database unreachable"

        except ImportError:
            health.add_check("psycopg2", False, "psycopg2 not installed")
            health.message = "psycopg2 missing"
        except Exception as e:
            health.add_check("database", False, f"Error: {str(e)}")
            health.message = "Database error"

        health.check_time = time.time() - start
        return health

    # ==========================================
    # Agent 3: Model Checker
    # ==========================================
    async def check_models(self) -> ComponentHealth:
        """Verifica que los modelos estén disponibles"""
        start = time.time()
        health = ComponentHealth("models")

        # Check Vision Model (ViT)
        try:
            import torch
            import torchvision.models as models
            from config import ModelConfig

            model_name = ModelConfig.VISION_MODEL
            if model_name == "vit_b_16":
                model = models.vit_b_16(pretrained=True)
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
            else:
                model = models.resnet50(pretrained=True)

            health.add_check(
                "vision_model",
                model is not None,
                f"{model_name} loaded successfully"
            )

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            health.add_check(
                "cuda",
                True,  # No es crítico
                f"CUDA available: {cuda_available}"
            )

        except Exception as e:
            health.add_check("vision_model", False, f"Error: {str(e)}")

        # Check Embedding Model
        try:
            from sentence_transformers import SentenceTransformer
            from config import ModelConfig

            embedding_model = SentenceTransformer(ModelConfig.EMBEDDING_MODEL)
            health.add_check(
                "embedding_model",
                embedding_model is not None,
                f"{ModelConfig.EMBEDDING_MODEL} loaded"
            )
        except Exception as e:
            health.add_check("embedding_model", False, f"Error: {str(e)}")

        # Check LLM API
        try:
            from openai import AsyncOpenAI
            from config import Config

            client = AsyncOpenAI(api_key=Config.api.OPENAI_API_KEY)
            health.add_check(
                "llm_client",
                client is not None,
                f"OpenAI client initialized"
            )
        except Exception as e:
            health.add_check("llm_client", False, f"Error: {str(e)}")

        health.message = "Models loaded"
        health.check_time = time.time() - start
        return health

    # ==========================================
    # Agent 4: Services Checker
    # ==========================================
    async def check_services(self) -> ComponentHealth:
        """Verifica servicios externos"""
        start = time.time()
        health = ComponentHealth("services")

        # Check Vector Service
        try:
            from agent_service import vector_service

            if vector_service:
                health.add_check(
                    "vector_service",
                    not vector_service.circuit_open,
                    "Vector service operational" if not vector_service.circuit_open else "Circuit breaker open"
                )
            else:
                health.add_check("vector_service", False, "Not initialized")
        except Exception as e:
            health.add_check("vector_service", False, f"Error: {str(e)}")

        # Check Web Search Service
        try:
            from agent_service import web_service

            if web_service:
                health.add_check(
                    "web_search_service",
                    not web_service.circuit_open,
                    "Web search operational" if not web_service.circuit_open else "Circuit breaker open"
                )
            else:
                health.add_check("web_search_service", False, "Not initialized")
        except Exception as e:
            health.add_check("web_search_service", False, f"Error: {str(e)}")

        # Check Notification Services
        try:
            from agent_service import email_service, webhook_service

            health.add_check(
                "email_service",
                email_service is not None,
                "Email service available" if email_service else "Not configured"
            )

            health.add_check(
                "webhook_service",
                webhook_service is not None,
                "Webhook service available" if webhook_service else "Not configured"
            )
        except Exception as e:
            health.add_check("notification_services", False, f"Error: {str(e)}")

        health.message = "Services running"
        health.check_time = time.time() - start
        return health

    # ==========================================
    # Agent 5: Agents Checker
    # ==========================================
    async def check_agents(self) -> ComponentHealth:
        """Verifica que todos los agentes estén disponibles"""
        start = time.time()
        health = ComponentHealth("agents")

        try:
            from agent_service import (
                query_classifier,
                rag_agent,
                web_search_agent,
                itinerary_agent,
                conversational_agent,
                build_assistant_graph
            )

            # Check cada agente
            agents = [
                ("classifier", query_classifier),
                ("rag", rag_agent),
                ("web_search", web_search_agent),
                ("itinerary", itinerary_agent),
                ("conversational", conversational_agent),
            ]

            for name, agent_func in agents:
                health.add_check(
                    f"{name}_agent",
                    agent_func is not None,
                    f"{name.title()} agent loaded"
                )

            # Check LangGraph
            try:
                graph = build_assistant_graph()
                health.add_check(
                    "langgraph",
                    graph is not None,
                    "LangGraph compiled successfully"
                )
            except Exception as e:
                health.add_check("langgraph", False, f"LangGraph error: {str(e)}")

            health.message = "5 agents operational"

        except Exception as e:
            health.add_check("agents", False, f"Error: {str(e)}")
            health.message = "Agent loading error"

        health.check_time = time.time() - start
        return health

    # ==========================================
    # Orchestrator: Run All Checks
    # ==========================================
    async def run_all_checks(self) -> Dict[str, Any]:
        """Ejecuta todos los health checks en paralelo"""
        logger.info("Running multi-agent health check...")

        start_time = time.time()

        # Ejecutar todos los checks en paralelo
        results = await asyncio.gather(
            self.check_configuration(),
            self.check_database(),
            self.check_models(),
            self.check_services(),
            self.check_agents(),
            return_exceptions=True
        )

        # Procesar resultados
        components = {}
        overall_status = HealthStatus.HEALTHY
        critical_failures = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                continue

            health: ComponentHealth = result
            components[health.name] = health.to_dict()

            # Determinar estado global
            if health.status == HealthStatus.UNHEALTHY:
                # Config y models son críticos
                if health.name in ["configuration", "models"]:
                    overall_status = HealthStatus.UNHEALTHY
                    critical_failures.append(health.name)
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        total_time = time.time() - start_time

        summary = {
            "status": overall_status.value,
            "timestamp": time.time(),
            "check_duration_ms": round(total_time * 1000, 2),
            "components": components,
            "summary": {
                "total_components": len(components),
                "healthy": len([c for c in components.values() if c["status"] == "healthy"]),
                "degraded": len([c for c in components.values() if c["status"] == "degraded"]),
                "unhealthy": len([c for c in components.values() if c["status"] == "unhealthy"]),
            }
        }

        if critical_failures:
            summary["critical_failures"] = critical_failures

        logger.info(f"Health check completed in {total_time:.2f}s - Status: {overall_status.value}")

        return summary


# Singleton global
health_checker = HealthChecker()
