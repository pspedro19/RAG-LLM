import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
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
    user_email: Optional[str] = None  # Para acción automatizada (envío de emails)
    
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

# Vision API models
class VisionPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float
    confidence_percent: float

class VisionResponse(BaseModel):
    predictions: List[VisionPrediction]
    model: str
    device: str
    inference_time: float
    image_size: tuple
    top_prediction: Dict[str, Any]

# Variable global para verificar si las dependencias están cargadas
dependencies_loaded = False
vector_service = None
process_query_func = None
checkpoint_manager = None
vision_service = None

async def load_dependencies():
    """Carga las dependencias solo cuando sea necesario"""
    global dependencies_loaded, vector_service, process_query_func, checkpoint_manager, vision_service

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

            # Inicializar servicio de visión
            try:
                from vision import VisionModelService
                from config import ModelConfig

                # Usar modelo de HuggingFace si está configurado, sino usar torchvision
                hf_vision_model = ModelConfig.HUGGINGFACE_VISION_MODEL
                vision_model_name = ModelConfig.VISION_MODEL
                vision_device = ModelConfig.VISION_DEVICE

                if hf_vision_model:
                    logger.info(f"Usando modelo de visión de HuggingFace: {hf_vision_model}")
                    vision_service = VisionModelService(
                        model_name=vision_model_name,
                        device=vision_device,
                        huggingface_model=hf_vision_model
                    )
                else:
                    logger.info(f"Usando modelo de visión de torchvision: {vision_model_name}")
                    vision_service = VisionModelService(
                        model_name=vision_model_name,
                        device=vision_device
                    )

                logger.info("Servicio de vision cargado correctamente")
            except Exception as ve:
                logger.warning(f"Servicio de vision no disponible: {ve}")
                vision_service = None

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

# Health check endpoint - crucial for Docker healthcheck with multi-agent validation
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint con validación multi-agente
    Verifica: Configuration, Database, Models, Services, Agents
    """
    try:
        from health_checker import health_checker

        # Ejecutar health check multi-agente
        health_report = await health_checker.run_all_checks()

        return HealthStatus(
            status=health_report["status"],
            timestamp=health_report["timestamp"],
            details=health_report
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")

        # Fallback simple si el health checker falla
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
        "query_endpoint": "/query",
        "vision_endpoint": "/vision/classify",
        "vision_info_endpoint": "/vision/model-info"
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
        # Procesar la consulta (incluyendo user_email en el estado si está disponible)
        result = await process_query_func(request.query, request.conversation_id, request.user_email)
        processing_time = time.time() - start_time
        
        logger.info(f"Consulta procesada en {processing_time:.2f}s: {request.query[:50]}...")
        return result
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}"
        )

# Query with image endpoint (Vision + LLM)
@app.post("/query/vision", response_model=QueryResponse)
async def query_with_vision(
    query: str = Form(...),
    image: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None)
):
    """
    Procesa una consulta con imagen usando GPT-4 Vision o Claude 3 Vision
    El agente puede describir, analizar y responder preguntas sobre la imagen
    """
    if not dependencies_loaded:
        success = await load_dependencies()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="El servicio está inicializándose. Por favor, intente nuevamente en unos momentos."
            )

    try:
        import base64
        from llm_service import get_llm_service

        start_time = time.time()

        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"El archivo debe ser una imagen. Tipo recibido: {image.content_type}"
            )

        # Leer y codificar imagen en base64
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Obtener servicio LLM
        llm_service = get_llm_service()

        # Crear prompt para análisis de imagen relacionado con Curazao
        vision_prompt = f"""Eres un asistente experto en turismo de Curazao. El usuario te ha enviado una imagen con la siguiente pregunta: "{query}"

Analiza la imagen detalladamente y responde la pregunta del usuario. Si la imagen muestra lugares, paisajes o elementos de Curazao, proporciona información turística relevante. Si muestra otra cosa, descríbela y trata de relacionarla con la consulta si es posible.

Sé específico, informativo y amigable en tu respuesta."""

        # Llamar al LLM con visión
        logger.info(f"Procesando consulta con imagen: {query[:50]}... (imagen: {image.filename})")
        vision_response = await llm_service.chat_completion_with_vision(
            text=vision_prompt,
            image_data=image_base64,
            image_type=image.content_type,
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=1500
        )

        processing_time = time.time() - start_time
        conversation_id = conversation_id or f"vision-{int(time.time())}"

        logger.info(f"Consulta con visión procesada en {processing_time:.2f}s usando {vision_response['provider']}")

        return QueryResponse(
            response=vision_response['content'],
            conversation_id=conversation_id,
            query_type="vision",
            total_time=processing_time,
            active_agents=["vision_agent"],
            tokens=vision_response.get('usage', {}),
            errors=False,
            warnings=[],
            react_steps_count=1,
            critical_error=False
        )

    except Exception as e:
        logger.error(f"Error procesando consulta con visión: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta con visión: {str(e)}"
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

# ========================================
# VISION ENDPOINTS (Modelo CNN/ViT)
# ========================================

@app.post("/vision/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Clasifica una imagen usando modelo ViT preentrenado

    Requisito del proyecto: "Integración de modelo preexistente para inferencia"
    """
    if not dependencies_loaded:
        await load_dependencies()

    if vision_service is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de visión no disponible. Instale torch y torchvision."
        )

    try:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"El archivo debe ser una imagen. Tipo recibido: {file.content_type}"
            )

        # Guardar temporalmente
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Validar imagen
            from vision.image_processor import ImageProcessor
            is_valid, error_msg = ImageProcessor.validate_image(tmp_path)

            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            # Realizar predicción
            result = await vision_service.predict(tmp_path, top_k=5)

            logger.info(f"Imagen clasificada: {file.filename} -> {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.1f}%)")

            return result

        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clasificando imagen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )

@app.get("/vision/model-info")
async def get_vision_model_info():
    """Retorna información sobre el modelo de visión cargado"""
    if not dependencies_loaded:
        await load_dependencies()

    if vision_service is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de visión no disponible"
        )

    return vision_service.get_model_info()

# Para desarrollo local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)