# video-processing-microservice/app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import asyncio
import httpx
from typing import Dict, Any, Optional

from app.config.settings import settings
from app.services.video_processor import VideoProcessor
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar servicios
video_processor = VideoProcessor()
embedding_service = EmbeddingService()
pinecone_service = PineconeService()

# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verificar API Key si está configurada"""
    if settings.API_KEY:
        if not credentials or credentials.credentials != settings.API_KEY:
            raise HTTPException(status_code=401, detail="API Key inválida")
    return True

# Crear aplicación
app = FastAPI(
    title="Video Processing Microservice",
    description="Microservicio especializado en procesamiento de videos con IA para clasificación de productos",
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes exactos
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check del microservicio"""
    
    embedding_health = await embedding_service.health_check()
    pinecone_health = await pinecone_service.health_check()
    
    return {
        "service": "Video Processing Microservice",
        "version": settings.VERSION,
        "status": "healthy" if embedding_health and pinecone_health else "degraded",
        "components": {
            "embedding_service": "healthy" if embedding_health else "degraded",
            "pinecone_service": "healthy" if pinecone_health else "degraded",
            "video_processor": "healthy"
        },
        "config": {
            "max_video_size_mb": settings.MAX_VIDEO_SIZE_MB,
            "max_frames": settings.MAX_FRAMES_TO_EXTRACT,
            "max_processing_time": settings.MAX_PROCESSING_TIME_MINUTES
        }
    }

@app.post("/api/v1/process-video")
async def process_video(
    job_id: int = Form(..., description="ID del job en sistema principal"),
    callback_url: str = Form(..., description="URL para notificar completación"),
    metadata: str = Form(..., description="Metadata del job en JSON"),
    video: UploadFile = File(..., description="Archivo de video a procesar"),
    _: bool = Depends(verify_api_key)
):
    """
    Endpoint principal para procesamiento de video
    """
    try:
        logger.info(f"🎬 Iniciando procesamiento - Job ID: {job_id}")
        
        # Validar video
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser un video")
        
        max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
        if video.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Video muy grande. Máximo: {settings.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Parsear metadata
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata debe ser JSON válido")
        
        # Procesar video (en background task)
        asyncio.create_task(
            process_video_background(job_id, video, metadata_dict, callback_url)
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Video recibido y en procesamiento",
            "estimated_time_minutes": "2-5"
        }
        
    except Exception as e:
        logger.error(f"❌ Error recibiendo video job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_background(
    job_id: int,
    video: UploadFile,
    metadata: Dict[str, Any],
    callback_url: str
):
    """Procesamiento de video en background"""
    try:
        logger.info(f"🔄 Procesamiento background iniciado - Job ID: {job_id}")
        
        # Procesar video
        results = await video_processor.process_video_complete(job_id, video, metadata)
        
        # Notificar al sistema principal
        await notify_completion(callback_url, job_id, "completed", results)
        
        logger.info(f"✅ Job {job_id} completado y notificado")
        
    except Exception as e:
        logger.error(f"❌ Error procesando job {job_id}: {e}")
        
        # Notificar error
        error_result = {
            "error_message": str(e),
            "job_id": job_id
        }
        await notify_completion(callback_url, job_id, "failed", error_result)

async def notify_completion(callback_url: str, job_id: int, status: str, results: Dict[str, Any]):
    """Notificar completación al sistema principal"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                callback_url,
                data={
                    "job_id": job_id,
                    "status": status,
                    "results": json.dumps(results)
                }
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Notificación enviada - Job ID: {job_id}")
            else:
                logger.error(f"❌ Error notificando job {job_id}: {response.status_code}")
                
    except Exception as e:
        logger.error(f"❌ Error enviando notificación job {job_id}: {e}")

@app.get("/api/v1/job-status/{job_id}")
async def get_job_status(
    job_id: int,
    _: bool = Depends(verify_api_key)
):
    """Consultar estado de job (placeholder - implementar con Redis/DB)"""
    # En producción, esto consultaría el estado real del job
    return {
        "job_id": job_id,
        "status": "processing",
        "progress_percentage": 50,
        "estimated_remaining_minutes": 2
    }

@app.get("/")
async def root():
    """Información del microservicio"""
    return {
        "service": "Video Processing Microservice",
        "version": settings.VERSION,
        "status": "running",
        "endpoints": {
            "process_video": "/api/v1/process-video",
            "job_status": "/api/v1/job-status/{job_id}",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )