# video-processing-microservice/app/main.py - VERSIÓN ACTUALIZADA
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Security , Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import asyncio
import httpx
import os
from typing import Dict, Any, Optional

from app.config.settings import settings

# Configurar logging (✅ mantener tu configuración)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security (✅ mantener igual)
security = HTTPBearer(auto_error=False)

async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """🔍 VERSIÓN CON LOGS DETALLADOS - Verificar API Key"""
    
    # 🔍 LOG: Configuración del microservicio
    microservice_api_key = getattr(settings, 'API_KEY', 'NO_CONFIGURADA')
    logger.info(f"🔧 MICROSERVICIO - API_KEY configurada: {microservice_api_key[:10] + '...' if microservice_api_key and microservice_api_key != 'NO_CONFIGURADA' else 'VACÍA/NO_CONFIGURADA'}")
    
    # 🔍 LOG: Headers recibidos
    auth_header = request.headers.get("Authorization")
    x_api_key_header = request.headers.get("X-API-Key")
    
    logger.info(f"📥 HEADERS RECIBIDOS:")
    logger.info(f"   - Authorization: {auth_header[:20] + '...' if auth_header else 'NO_PRESENTE'}")
    logger.info(f"   - X-API-Key: {x_api_key_header[:10] + '...' if x_api_key_header else 'NO_PRESENTE'}")
    
    # 🔍 LOG: Credentials del Security
    if credentials:
        logger.info(f"📋 CREDENTIALS - scheme: {credentials.scheme}, credentials: {credentials.credentials[:10] + '...' if credentials.credentials else 'VACÍO'}")
    else:
        logger.info(f"📋 CREDENTIALS: None")
    
    # 🔍 VALIDACIÓN: Si no hay API_KEY configurada, permitir acceso
    if not settings.API_KEY:
        logger.info("✅ ACCESO PERMITIDO - No hay API_KEY configurada en el microservicio")
        return True
    
    # 🔍 VALIDACIÓN: Verificar diferentes métodos de autenticación
    received_api_key = None
    auth_method = None
    
    # Método 1: X-API-Key header (preferido)
    if x_api_key_header:
        received_api_key = x_api_key_header
        auth_method = "X-API-Key header"
    
    # Método 2: Authorization Bearer
    elif credentials and credentials.credentials:
        received_api_key = credentials.credentials
        auth_method = "Authorization Bearer"
    
    # Método 3: Authorization header directo
    elif auth_header and auth_header.startswith("Bearer "):
        received_api_key = auth_header.replace("Bearer ", "")
        auth_method = "Authorization header directo"
    
    # 🔍 LOG: Comparación detallada
    logger.info(f"🔑 COMPARACIÓN DE API KEYS:")
    logger.info(f"   - Método usado: {auth_method or 'NINGUNO'}")
    logger.info(f"   - Key recibida: {received_api_key[:10] + '...' if received_api_key else 'NO_RECIBIDA'}")
    logger.info(f"   - Key esperada: {settings.API_KEY[:10] + '...' if settings.API_KEY else 'NO_CONFIGURADA'}")
    logger.info(f"   - ¿Coinciden?: {received_api_key == settings.API_KEY if received_api_key and settings.API_KEY else 'NO_COMPARABLE'}")
    
    # 🔍 VALIDACIÓN FINAL
    if not received_api_key:
        logger.error("❌ ACCESO DENEGADO - No se recibió API Key por ningún método")
        raise HTTPException(status_code=401, detail="API Key requerida")
    
    if received_api_key != settings.API_KEY:
        logger.error("❌ ACCESO DENEGADO - API Key no coincide")
        logger.error(f"   - Esperada: '{settings.API_KEY}' (longitud: {len(settings.API_KEY) if settings.API_KEY else 0})")
        logger.error(f"   - Recibida: '{received_api_key}' (longitud: {len(received_api_key)})")
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    logger.info("✅ ACCESO PERMITIDO - API Key válida")
    return True

# 🆕 AGREGAR: Lifespan para setup de credenciales (patrón del microservicio clasificación)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management con setup de credenciales Google Cloud"""
    logger.info("🚀 Iniciando Video Processing Microservice")
    
    try:
        # 🆕 SETUP DE CREDENCIALES (del microservicio clasificación)
        from app.config.google_auth import setup_google_credentials
        credentials_ok = setup_google_credentials()
        
        if credentials_ok:
            logger.info("✅ Credenciales Google Cloud configuradas")
        else:
            logger.warning("⚠️ Credenciales Google Cloud no disponibles")
        
        # ✅ MANTENER: Setup de directorio temporal
        os.makedirs(settings.TEMP_STORAGE_PATH, exist_ok=True)
        logger.info("✅ Directorio temporal creado")
        
        # ✅ MANTENER: Verificar health de servicios al startup (opcional)
        try:
            # Solo importar después de configurar credenciales
            from app.services.embedding_service import EmbeddingService
            from app.services.pinecone_service import PineconeService
            
            # Test rápido de servicios (no bloquear startup)
            embedding_service = EmbeddingService()
            pinecone_service = PineconeService()
            
            # Test health con timeout corto
            health_tasks = [
                asyncio.wait_for(embedding_service.health_check(), timeout=5.0),
                asyncio.wait_for(pinecone_service.health_check(), timeout=5.0)
            ]
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            embedding_ok = health_results[0] if not isinstance(health_results[0], Exception) else False
            pinecone_ok = health_results[1] if not isinstance(health_results[1], Exception) else False
            
            if embedding_ok:
                logger.info("✅ Google Multimodal - Conectado")
            else:
                logger.warning("⚠️ Google Multimodal - No disponible")
                
            if pinecone_ok:
                logger.info("✅ Pinecone - Conectado")
            else:
                logger.warning("⚠️ Pinecone - No disponible")
                
        except Exception as e:
            logger.warning(f"⚠️ Error en health checks de startup: {e}")
        
        logger.info("🟢 Microservicio listo para recibir requests")
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        # No fallar el startup, continuar
    
    yield
    
    # Cleanup
    logger.info("🔄 Cerrando Video Processing Microservice")

# ✅ ACTUALIZAR: Usar lifespan en FastAPI
app = FastAPI(
    title="Video Processing Microservice",
    description="Microservicio especializado en procesamiento de videos con IA para clasificación de productos",
    version=settings.VERSION,
    lifespan=lifespan,  # 🆕 AGREGAR lifespan
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# ✅ MANTENER: CORS igual
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 🔄 ACTUALIZAR: Inicializar servicios como variables globales (después de credentials)
video_processor = None
embedding_service = None
pinecone_service = None

def get_services():
    """Obtener servicios inicializados (lazy loading)"""
    global video_processor, embedding_service, pinecone_service
    
    if video_processor is None:
        from app.services.video_processor import VideoProcessor
        from app.services.embedding_service import EmbeddingService
        from app.services.pinecone_service import PineconeService
        
        video_processor = VideoProcessor()
        embedding_service = EmbeddingService()
        pinecone_service = PineconeService()
    
    return video_processor, embedding_service, pinecone_service

# ✅ MANTENER: Health check (actualizar para usar lazy loading)
@app.get("/health")
async def health_check():
    """Health check del microservicio"""
    try:
        _, embedding_service, pinecone_service = get_services()
        
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
                "max_processing_time": settings.MAX_PROCESSING_TIME_MINUTES,
                "credentials_configured": bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
            }
        }
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return {
            "service": "Video Processing Microservice",
            "version": settings.VERSION,
            "status": "degraded",
            "error": str(e)
        }

# ✅ MANTENER: Endpoint de procesamiento (actualizar para usar lazy loading)
# video-processing-microservice/app/main.py - ENDPOINT SIMPLIFICADO
@app.post("/api/v1/process-video")
async def process_video(
    request: Request,
    job_id: int = Form(..., description="ID del job en sistema principal"),
    callback_url: str = Form(..., description="URL para notificar completación"),
    metadata: str = Form(..., description="Metadata del job en JSON"),
    video: UploadFile = File(..., description="Archivo de video a procesar")
    # ❌ REMOVER: _: bool = Depends(lambda req=request: verify_api_key(req, Security(security)))
):
    """🎬 ENDPOINT SIMPLIFICADO SIN DEPENDENCY COMPLEJO"""
    
    # ✅ LLAMAR DIRECTAMENTE A LA VALIDACIÓN
    try:
        await verify_api_key_with_logs(request)
    except HTTPException as e:
        # Re-lanzar la excepción de autenticación
        raise e
    
    try:
        logger.info(f"🎬 INICIANDO PROCESAMIENTO - Job ID: {job_id}")
        logger.info(f"📞 Callback URL: {callback_url}")
        logger.info(f"📊 Metadata: {metadata}")
        logger.info(f"📹 Video: {video.filename}, Size: {video.size} bytes")
        
        # Validaciones básicas
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser un video")
        
        max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
        if video.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Video muy grande. Máximo: {settings.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Por ahora, responder con éxito simulado
        logger.info(f"✅ PROCESAMIENTO SIMULADO COMPLETADO - Job ID: {job_id}")
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Video recibido y procesando",
            "estimated_time_minutes": 2
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR EN PROCESAMIENTO Job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# video-processing-microservice/app/main.py - FUNCIÓN SIMPLIFICADA
async def verify_api_key_with_logs(request: Request):
    """🔍 VALIDACIÓN SIMPLIFICADA CON LOGS DETALLADOS"""
    
    # 🔍 LOG: Configuración del microservicio
    microservice_api_key = getattr(settings, 'API_KEY', None)
    logger.info(f"🔧 MICROSERVICIO - API_KEY configurada: {'SÍ' if microservice_api_key else 'NO'}")
    if microservice_api_key:
        logger.info(f"🔧 MICROSERVICIO - Primeros 10 chars: {microservice_api_key[:10]}...")
    
    # 🔍 LOG: Headers recibidos
    auth_header = request.headers.get("Authorization", "")
    x_api_key_header = request.headers.get("X-API-Key", "")
    
    logger.info(f"📥 HEADERS RECIBIDOS:")
    logger.info(f"   - Authorization: {'SÍ' if auth_header else 'NO'}")
    logger.info(f"   - X-API-Key: {'SÍ' if x_api_key_header else 'NO'}")
    
    if auth_header:
        logger.info(f"   - Authorization value: {auth_header[:20]}...")
    if x_api_key_header:
        logger.info(f"   - X-API-Key value: {x_api_key_header[:10]}...")
    
    # 🔍 VALIDACIÓN: Si no hay API_KEY configurada, permitir acceso
    if not settings.API_KEY:
        logger.info("✅ ACCESO PERMITIDO - No hay API_KEY configurada")
        return True
    
    # 🔍 OBTENER API KEY del request
    received_api_key = None
    auth_method = None
    
    if x_api_key_header:
        received_api_key = x_api_key_header
        auth_method = "X-API-Key header"
    elif auth_header.startswith("Bearer "):
        received_api_key = auth_header.replace("Bearer ", "")
        auth_method = "Authorization Bearer"
    
    # 🔍 LOG: Comparación
    logger.info(f"🔑 COMPARACIÓN:")
    logger.info(f"   - Método: {auth_method or 'NINGUNO'}")
    logger.info(f"   - Key recibida: {'SÍ' if received_api_key else 'NO'}")
    logger.info(f"   - Coinciden: {received_api_key == settings.API_KEY if received_api_key and settings.API_KEY else 'N/A'}")
    
    # 🔍 VALIDACIÓN FINAL
    if not received_api_key:
        logger.error("❌ ACCESO DENEGADO - No hay API Key")
        raise HTTPException(status_code=401, detail="API Key requerida")
    
    if received_api_key != settings.API_KEY:
        logger.error("❌ ACCESO DENEGADO - API Key no coincide")
        logger.error(f"   - Esperada: '{settings.API_KEY}'")
        logger.error(f"   - Recibida: '{received_api_key}'")
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    logger.info("✅ ACCESO PERMITIDO - API Key válida")
    return True

# ✅ MANTENER: Procesamiento background (actualizar para usar lazy loading)
async def process_video_background(
    job_id: int,
    video: UploadFile,
    metadata: Dict[str, Any],
    callback_url: str
):
    """✅ MANTENER: Procesamiento de video en background"""
    try:
        logger.info(f"🔄 Procesamiento background iniciado - Job ID: {job_id}")
        
        # 🔄 ACTUALIZAR: Obtener servicios con lazy loading
        video_processor, _, _ = get_services()
        
        # ✅ MANTENER: Procesar video igual
        results = await video_processor.process_video_complete(job_id, video, metadata)
        
        # ✅ MANTENER: Notificar igual
        await notify_completion(callback_url, job_id, "completed", results)
        
        logger.info(f"✅ Job {job_id} completado y notificado")
        
    except Exception as e:
        logger.error(f"❌ Error procesando job {job_id}: {e}")
        
        error_result = {
            "error_message": str(e),
            "job_id": job_id
        }
        await notify_completion(callback_url, job_id, "failed", error_result)

# ✅ MANTENER: Notificación igual
async def notify_completion(callback_url: str, job_id: int, status: str, results: Dict[str, Any]):
    """
    Notificar completación al sistema principal - MEJORADO
    """
    try:
        # 🆕 AGREGAR MÁS METADATOS EN EL CALLBACK
        callback_data = {
            "job_id": job_id,
            "status": status,
            "results": json.dumps({
                **results,
                "callback_timestamp": datetime.now().isoformat(),
                "microservice_version": settings.VERSION
            })
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(callback_url, data=callback_data)
            
            if response.status_code == 200:
                logger.info(f"✅ Callback enviado exitosamente - Job ID: {job_id}")
            else:
                logger.error(f"❌ Error en callback job {job_id}: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ Error enviando callback job {job_id}: {e}")

        
# ✅ MANTENER: Job status igual
@app.get("/api/v1/job-status/{job_id}")
async def get_job_status(
    job_id: int,
    _: bool = Depends(verify_api_key)
):
    """✅ MANTENER: Consultar estado de job"""
    return {
        "job_id": job_id,
        "status": "processing",
        "progress_percentage": 50,
        "estimated_remaining_minutes": 2
    }

# ✅ MANTENER: Root endpoint igual
@app.get("/")
async def root():
    """✅ MANTENER: Información del microservicio"""
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

# ✅ MANTENER: Main igual
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )