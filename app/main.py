import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fastapi-app")

# Create FastAPI app
app = FastAPI(
    title="Curacao Tourism Assistant API",
    description="API for a multi-agent tourism assistant focused on Curacao",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API models for request/response
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    
class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    query_type: Optional[str] = None
    total_time: float
    active_agents: Optional[List[str]] = Field(default_factory=list)
    tokens: Optional[Dict[str, int]] = None
    errors: Optional[bool] = False
    warnings: Optional[List[str]] = Field(default_factory=list)
    react_steps_count: Optional[int] = None
    critical_error: Optional[bool] = False

# Health check model for detailed health status    
class HealthStatus(BaseModel):
    status: str
    timestamp: float
    details: Dict[str, Any]

# Variable global para verificar si las dependencias están cargadas
dependencies_loaded = False
vector_service = None 
process_query_func = None
checkpoint_manager = None

async def load_dependencies():
    """Carga las dependencias solo cuando sea necesario"""
    global dependencies_loaded, vector_service, process_query_func, checkpoint_manager
    
    if dependencies_loaded:
        return True
        
    try:
        logger.info("Cargando dependencias...")
        # Intentar importar las dependencias
        try:
            import agent_service
            from agent_service import process_query, SimpleCheckpointManager, validate_env, vector_service as vs
            
            # Asignar a las variables globales
            process_query_func = process_query
            vector_service = vs
            checkpoint_manager = SimpleCheckpointManager("./checkpoints")
            
            # Validar variables de entorno críticas
            critical_vars = ["OPENAI_API_KEY"]
            if not validate_env(critical_vars):
                logger.error("Variables de entorno críticas faltantes")
                return False
                
            # Inicializar el servicio vectorial
            await vector_service.initialize()
            logger.info("Dependencias cargadas correctamente")
            dependencies_loaded = True
            return True
            
        except ImportError as e:
            logger.error(f"Error importando dependencias: {e}")
            return False
    except Exception as e:
        logger.error(f"Error cargando dependencias: {e}")
        return False

# Startup event para verificar dependencias sin bloquear el inicio
@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando API...")
    # No bloqueamos el inicio, pero intentamos precargar las dependencias
    asyncio.create_task(load_dependencies())

# Health check endpoint - crucial for Docker healthcheck
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint for the API"""
    if not dependencies_loaded:
        await load_dependencies()
        
    details = {
        "api_status": "operational",
        "dependencies_loaded": dependencies_loaded
    }
    
    # Si las dependencias están cargadas, verificar el estado del servicio vectorial
    if dependencies_loaded and vector_service:
        details["vector_db_status"] = "healthy" if not vector_service.circuit_open else "degraded"
        details["vector_db_fallback_mode"] = vector_service.fallback_mode
        
        # Intentar inicializar si es necesario
        try:
            if not await vector_service.initialize():
                details["vector_db_status"] = "initializing"
        except Exception as e:
            details["vector_db_status"] = "error"
            details["vector_db_error"] = str(e)
    else:
        details["vector_db_status"] = "not_loaded"
    
    # Determinar el estado general
    if not dependencies_loaded:
        status = "initializing"
    elif details.get("vector_db_status") in ["healthy", "degraded"]:
        status = "healthy"
    else:
        # Aunque la base de datos vectorial no funcione, consideramos el servicio
        # como "degraded" en lugar de "unhealthy" para pasar health checks de Docker
        status = "degraded"
    
    return HealthStatus(
        status=status,
        timestamp=time.time(),
        details=details
    )

# Endpoint simple para pruebas
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "api": "Curacao Tourism Assistant",
        "version": "1.0.0",
        "health_endpoint": "/health",
        "query_endpoint": "/query"
    }

# Main endpoint for processing queries
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query through the multi-agent system"""
    # Cargar las dependencias si aún no se han cargado
    if not dependencies_loaded:
        success = await load_dependencies()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="El servicio está inicializándose. Por favor, intente nuevamente en unos momentos."
            )
    
    try:
        start_time = time.time()
        # Procesar la consulta
        result = await process_query_func(request.query, request.conversation_id)
        processing_time = time.time() - start_time
        
        logger.info(f"Consulta procesada en {processing_time:.2f}s: {request.query[:50]}...")
        return result
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}"
        )

# Conversation history endpoint
@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get the conversation history for a specific conversation ID"""
    # Cargar las dependencias si aún no se han cargado
    if not dependencies_loaded:
        success = await load_dependencies()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="El servicio está inicializándose. Por favor, intente nuevamente en unos momentos."
            )
    
    try:
        checkpoint = checkpoint_manager.get(conversation_id)
        if not checkpoint or "conversation_history" not in checkpoint:
            return {"conversation_id": conversation_id, "messages": []}
        
        return {
            "conversation_id": conversation_id,
            "messages": checkpoint["conversation_history"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving conversation history: {str(e)}"
        )

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)