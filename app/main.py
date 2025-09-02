# video-processing-microservice/app/main.py - VERSIÓN ULTRA-OPTIMIZADA
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse 
import json
import logging
import asyncio
import httpx
import os
from typing import Dict, Any
from datetime import datetime
import time

from app.config.settings import settings

# Configurar logging optimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting para requests
request_times = {}

async def verify_api_key_optimized(request: Request):
    """🔐 Verificación de API Key optimizada"""
    
    # Si no hay API_KEY configurada, permitir acceso
    if not settings.API_KEY:
        return True
    
    # Obtener API key de headers
    api_key = request.headers.get("X-API-Key") or \
             request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida o faltante")
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """🚀 Lifecycle management ultra-optimizado"""
    
    logger.info("🚀 Iniciando Video Processing Microservice Ultra-Optimizado")
    
    try:
        # Setup de credenciales Google Cloud
        from app.config.google_auth import setup_google_credentials
        credentials_ok = setup_google_credentials()
        
        if credentials_ok:
            logger.info("✅ Credenciales Google Cloud configuradas")
        else:
            logger.warning("⚠️ Credenciales Google Cloud no disponibles")
        
        # Crear directorio temporal
        os.makedirs(settings.TEMP_STORAGE_PATH, exist_ok=True)
        logger.info("✅ Directorio temporal creado")
        
        # Pre-inicializar servicios optimizados
        from app.services.video_processor import video_processor_optimized
        from app.services.embedding_service_optimized import embedding_service_optimized
        from app.services.pinecone_service import PineconeService
        
        # Health check no-bloqueante al startup
        asyncio.create_task(startup_health_check())
        
        logger.info("🟢 Microservicio ultra-optimizado listo")
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
    
    yield
    
    logger.info("🔄 Cerrando microservicio")

async def startup_health_check():
    """🏥 Health check de startup no-bloqueante"""
    try:
        await asyncio.sleep(2)  # Dar tiempo a la inicialización
        
        from app.services.embedding_service_optimized import embedding_service_optimized
        from app.services.pinecone_service import PineconeService
        
        pinecone_service = PineconeService()
        
        # Tests con timeout
        embedding_ok = await asyncio.wait_for(
            embedding_service_optimized.health_check(), timeout=10.0
        )
        pinecone_ok = await asyncio.wait_for(
            pinecone_service.health_check(), timeout=5.0
        )
        
        logger.info(f"🏥 Health check startup: Embedding={'✅' if embedding_ok else '❌'}, Pinecone={'✅' if pinecone_ok else '❌'}")
        
    except Exception as e:
        logger.warning(f"⚠️ Health check startup falló: {e}")

# Crear app con lifespan optimizado
app = FastAPI(
    title="Video Processing Microservice Ultra-Optimized",
    description="Microservicio ultra-optimizado para procesamiento de videos cortos (3-5s) con IA",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS optimizado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time", "X-Strategy-Used"]
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """⏱️ Middleware para medir tiempo de procesamiento"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Processing-Time"] = str(round(process_time, 3))
    return response

@app.get("/health")
async def health_check_optimized():
    """🏥 Health check ultra-completo"""
    try:
        from app.services.embedding_service_optimized import embedding_service_optimized
        from app.services.pinecone_service import PineconeService
        
        pinecone_service = PineconeService()
        
        # Health checks con timeout
        health_tasks = [
            asyncio.wait_for(embedding_service_optimized.health_check(), timeout=5.0),
            asyncio.wait_for(pinecone_service.health_check(), timeout=3.0)
        ]
        
        embedding_health, pinecone_health = await asyncio.gather(
            *health_tasks, return_exceptions=True
        )
        
        embedding_ok = embedding_health if not isinstance(embedding_health, Exception) else False
        pinecone_ok = pinecone_health if not isinstance(pinecone_health, Exception) else False
        
        # Estadísticas del embedding service
        embedding_stats = embedding_service_optimized.get_statistics()
        
        # Determinar estado general
        if embedding_ok and pinecone_ok:
            status = "healthy"
        elif embedding_ok or pinecone_ok:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "service": "Video Processing Microservice Ultra-Optimized",
            "version": settings.VERSION,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "embedding_service": "healthy" if embedding_ok else "degraded",
                "pinecone_service": "healthy" if pinecone_ok else "degraded",
                "video_processor": "healthy"
            },
            "configuration": {
                "max_video_size_mb": settings.MAX_VIDEO_SIZE_MB,
                "max_frames_very_short": settings.MAX_FRAMES_VERY_SHORT_VIDEO,
                "max_frames_short": settings.MAX_FRAMES_SHORT_VIDEO,
                "max_frames_standard": settings.MAX_FRAMES_TO_EXTRACT,
                "parallel_processing": settings.ENABLE_PARALLEL_PROCESSING,
                "batch_size": settings.BATCH_SIZE_EMBEDDINGS,
                "optimization_level": "ultra_optimized"
            },
            "performance": {
                "embeddings_generated": embedding_stats.get("embeddings_generated", 0),
                "error_rate": embedding_stats.get("error_rate", 0),
                "rate_limiting_active": True
            },
            "thresholds": {
                "very_short_video_seconds": settings.VERY_SHORT_VIDEO_THRESHOLD_SECONDS,
                "short_video_seconds": settings.SHORT_VIDEO_THRESHOLD_SECONDS,
                "min_confidence_training": settings.MIN_CONFIDENCE_FOR_TRAINING,
                "min_confidence_response": settings.MIN_CONFIDENCE_FOR_RESPONSE
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return {
            "service": "Video Processing Microservice Ultra-Optimized",
            "version": settings.VERSION,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/v1/process-video")
async def process_video_ultra_optimized(
    request: Request,
    job_id: int = Form(..., description="ID del job en sistema principal"),
    callback_url: str = Form(..., description="URL para notificar completación"),
    metadata: str = Form(..., description="Metadata del job en JSON"),
    video: UploadFile = File(..., description="Archivo de video a procesar")
):
    """🎬 Endpoint ultra-optimizado para procesamiento de videos cortos"""
    
    start_time = time.time()
    
    try:
        # Verificar API Key
        await verify_api_key_optimized(request)
        
        # Rate limiting básico por IP
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip in request_times:
            if current_time - request_times[client_ip] < 1.0:  # 1 request per
                raise HTTPException(status_code=429, detail="Rate limit excedido. Máximo 1 request por segundo.")
       
        request_times[client_ip] = current_time
       
        logger.info(f"🎬 PROCESAMIENTO ULTRA-OPTIMIZADO INICIADO - Job ID: {job_id}")
        logger.info(f"📞 Callback: {callback_url}")
        logger.info(f"📹 Video: {video.filename}, Tamaño: {video.size} bytes")
       
       # Validaciones rápidas
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un video")
        
        max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
        if video.size and video.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Video muy grande. Máximo: {settings.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Parsear metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata JSON inválido")
        
        # Procesar video en background con processor optimizado
        asyncio.create_task(process_video_background_optimized(
            job_id, video, metadata_dict, callback_url
        ))
        
        processing_setup_time = time.time() - start_time
        
        response_data = {
            "status": "processing",
            "job_id": job_id,
            "message": "Video recibido y procesando con optimizaciones ultra-avanzadas",
            "estimated_time_seconds": 12,  # Estimación optimizada
            "optimization_level": "ultra_optimized",
            "setup_time_seconds": round(processing_setup_time, 3)
        }
        
        # Headers de respuesta con información de procesamiento
        response = JSONResponse(content=response_data)
        response.headers["X-Strategy-Used"] = "ultra_optimized"
        response.headers["X-Estimated-Time"] = "12"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ERROR PROCESAMIENTO Job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

async def process_video_background_optimized(
   job_id: int,
   video: UploadFile,
   metadata: Dict[str, Any],
   callback_url: str
):
   """🔄 Procesamiento de video en background ultra-optimizado"""
   
   start_time = time.time()
   
   try:
       logger.info(f"🔄 Procesamiento background ultra-optimizado iniciado - Job ID: {job_id}")
       
       # Usar el processor optimizado
       from app.services.video_processor import video_processor_optimized
       
       # Procesar con todas las optimizaciones
       results = await video_processor_optimized.process_video_complete(job_id, video, metadata)
       
       processing_time = time.time() - start_time
       results["total_processing_time_seconds"] = round(processing_time, 2)
       
       # Notificar con resultados enriquecidos
       await notify_completion_optimized(callback_url, job_id, "completed", results)
       
       logger.info(f"✅ Job {job_id} completado en {processing_time:.2f}s y notificado")
       
   except Exception as e:
       processing_time = time.time() - start_time
       logger.error(f"❌ Error procesando job {job_id} después de {processing_time:.2f}s: {e}")
       
       error_result = {
           "error_message": str(e),
           "job_id": job_id,
           "processing_time_seconds": round(processing_time, 2),
           "error_type": "processing_error",
           "optimization_level": "ultra_optimized"
       }
       
       await notify_completion_optimized(callback_url, job_id, "failed", error_result)

async def notify_completion_optimized(
   callback_url: str, 
   job_id: int, 
   status: str, 
   results: Dict[str, Any]
):
   """📞 Notificación optimizada con retry logic"""
   
   max_attempts = 3
   
   for attempt in range(max_attempts):
       try:
           # Preparar datos de callback enriquecidos
           callback_data = {
               "job_id": job_id,
               "status": status,
               "results": json.dumps({
                   **results,
                   "callback_timestamp": datetime.now().isoformat(),
                   "microservice_version": settings.VERSION,
                   "optimization_level": "ultra_optimized",
                   "attempt": attempt + 1
               }, ensure_ascii=False),
               "microservice_info": {
                   "version": settings.VERSION,
                   "optimization_level": "ultra_optimized",
                   "processing_node": "render_optimized"
               }
           }
           
           async with httpx.AsyncClient(timeout=30) as client:
               response = await client.post(
                   callback_url, 
                   data=callback_data,
                   headers={
                       "Content-Type": "application/x-www-form-urlencoded",
                       "X-Microservice-Version": settings.VERSION,
                       "X-Optimization-Level": "ultra_optimized"
                   }
               )
               
               if response.status_code == 200:
                   logger.info(f"✅ Callback enviado exitosamente - Job ID: {job_id} (intento {attempt + 1})")
                   return
               else:
                   raise httpx.HTTPError(f"HTTP {response.status_code}: {response.text}")
                   
       except Exception as e:
           if attempt < max_attempts - 1:
               wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
               logger.warning(f"⚠️ Callback intento {attempt + 1} falló: {e}. Reintentando en {wait_time}s...")
               await asyncio.sleep(wait_time)
           else:
               logger.error(f"❌ Error enviando callback job {job_id} después de {max_attempts} intentos: {e}")

@app.get("/api/v1/job-status/{job_id}")
async def get_job_status_optimized(job_id: int, request: Request):
   """📊 Estado de job con información optimizada"""
   
   await verify_api_key_optimized(request)
   
   # En una implementación real, consultarías una base de datos o cache
   # Por ahora, devolvemos estado simulado con información optimizada
   return {
       "job_id": job_id,
       "status": "processing",
       "optimization_level": "ultra_optimized",
       "estimated_completion_time": datetime.now().isoformat(),
       "progress": {
           "stage": "frame_processing",
           "percentage": 75,
           "current_operation": "generating_embeddings_batch"
       },
       "performance": {
           "strategy_being_used": "ultra_dense_sampling",
           "frames_processed": 6,
           "estimated_remaining_seconds": 3
       }
   }

@app.get("/api/v1/stats")
async def get_service_stats(request: Request):
   """📊 Estadísticas del servicio ultra-optimizado"""
   
   await verify_api_key_optimized(request)
   
   try:
       from app.services.embedding_service_optimized import embedding_service_optimized
       
       embedding_stats = embedding_service_optimized.get_statistics()
       
       return {
           "service": "Video Processing Microservice Ultra-Optimized",
           "version": settings.VERSION,
           "optimization_level": "ultra_optimized",
           "uptime_info": {
               "status": "running",
               "optimization_features": [
                   "adaptive_frame_extraction",
                   "parallel_processing",
                   "batch_embeddings",
                   "statistical_consolidation",
                   "intelligent_training"
               ]
           },
           "embedding_service": embedding_stats,
           "configuration": {
               "strategies": {
                   "very_short_videos": f"≤{settings.VERY_SHORT_VIDEO_THRESHOLD_SECONDS}s → ultra_dense_sampling",
                   "short_videos": f"≤{settings.SHORT_VIDEO_THRESHOLD_SECONDS}s → balanced_dense_sampling", 
                   "standard_videos": f">{settings.SHORT_VIDEO_THRESHOLD_SECONDS}s → quality_focused_sampling"
               },
               "performance_settings": {
                   "parallel_processing": settings.ENABLE_PARALLEL_PROCESSING,
                   "batch_size": settings.BATCH_SIZE_EMBEDDINGS,
                   "max_concurrent_frames": settings.MAX_CONCURRENT_FRAMES,
                   "frame_resize_target": settings.FRAME_RESIZE_TARGET
               },
               "quality_thresholds": {
                   "training": settings.MIN_CONFIDENCE_FOR_TRAINING,
                   "response": settings.MIN_CONFIDENCE_FOR_RESPONSE
               }
           }
       }
       
   except Exception as e:
       logger.error(f"❌ Error obteniendo estadísticas: {e}")
       return {"error": "Error obteniendo estadísticas del servicio"}

@app.get("/")
async def root_optimized():
   """🏠 Información del microservicio ultra-optimizado"""
   return {
       "service": "Video Processing Microservice Ultra-Optimized",
       "version": settings.VERSION,
       "optimization_level": "ultra_optimized",
       "status": "running",
       "specialization": "Videos cortos 3-5 segundos",
       "key_features": [
           "Extracción inteligente de frames con filtros adaptativos",
           "Procesamiento paralelo y por lotes",
           "Consolidación estadística avanzada",
           "Configuración adaptativa por duración de video",
           "Rate limiting y retry logic inteligente",
           "Health checks optimizados"
       ],
       "endpoints": {
           "process_video": "POST /api/v1/process-video",
           "job_status": "GET /api/v1/job-status/{job_id}",
           "health": "GET /health",
           "stats": "GET /api/v1/stats"
       },
       "performance": {
           "expected_processing_time": "12-15 segundos",
           "supported_video_lengths": "0.5-180 segundos",
           "optimal_range": "3-5 segundos",
           "max_concurrent_requests": "Limitado por recursos del servidor"
       }
   }

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
   logger.error(f"❌ Error no manejado en {request.url.path}: {exc}")
   return JSONResponse(
       status_code=500,
       content={
           "error": "Error interno del servidor",
           "detail": str(exc) if settings.DEBUG else "Error de procesamiento",
           "service": "Video Processing Microservice Ultra-Optimized",
           "path": str(request.url.path)
       }
   )

if __name__ == "__main__":
   import uvicorn
   
   uvicorn.run(
       "app.main:app",
       host=settings.HOST,
       port=settings.PORT,
       reload=settings.DEBUG,
       log_level="info",
       access_log=True,
       workers=1  # Single worker para manejo óptimo de recursos en Render
   )