# video-processing-microservice/app/config/settings.py - VERSIÓN OPTIMIZADA COMPLETA
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Dict, Any
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Video Processing Microservice - Optimized"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    API_KEY: Optional[str] = Field(default=None, description="API Key para autenticación")
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT_ID: str = Field(..., description="Google Cloud Project ID")
    GOOGLE_CLOUD_LOCATION: str = Field(default="us-central1", description="Google Cloud Location")
    
    # Pinecone
    PINECONE_API_KEY: str = Field(..., description="Pinecone API Key")
    PINECONE_INDEX_NAME: str = Field(default="sneaker-embeddings", description="Pinecone Index Name")
    
    # 🆕 CONFIGURACIÓN OPTIMIZADA PARA VIDEOS CORTOS (3-5 segundos)
    SHORT_VIDEO_THRESHOLD_SECONDS: float = Field(default=6.0, description="Umbral para videos cortos")
    VERY_SHORT_VIDEO_THRESHOLD_SECONDS: float = Field(default=3.0, description="Umbral para videos muy cortos")
    
    # Extracción de frames optimizada
    DENSE_SAMPLING_INTERVAL_SECONDS: float = Field(default=0.4, description="Intervalo denso para videos muy cortos")
    BALANCED_SAMPLING_INTERVAL_SECONDS: float = Field(default=0.6, description="Intervalo balanceado para videos cortos")
    STANDARD_SAMPLING_INTERVAL_SECONDS: float = Field(default=1.0, description="Intervalo estándar")
    
    # Límites adaptativos de frames
    MAX_FRAMES_VERY_SHORT_VIDEO: int = Field(default=8, description="Max frames para videos ≤3s")
    MAX_FRAMES_SHORT_VIDEO: int = Field(default=10, description="Max frames para videos 3-6s")
    MAX_FRAMES_TO_EXTRACT: int = Field(default=12, description="Max frames para videos >6s")
    
    # 🆕 FILTROS DE CALIDAD ADAPTATIVOS
    QUALITY_FILTERS: Dict[str, Dict[str, Any]] = Field(default={
        "very_short": {  # Videos ≤3 segundos - MUY PERMISIVO
            "min_brightness": 8,
            "min_variance": 25,
            "quality_threshold": 0.25,
            "focus_weight": 0.40,
            "brightness_weight": 0.15,
            "contrast_weight": 0.25,
            "motion_weight": 0.20
        },
        "short": {  # Videos 3-6 segundos - PERMISIVO
            "min_brightness": 12,
            "min_variance": 35,
            "quality_threshold": 0.35,
            "focus_weight": 0.40,
            "brightness_weight": 0.15,
            "contrast_weight": 0.25,
            "motion_weight": 0.20
        },
        "standard": {  # Videos >6 segundos - ESTRICTO
            "min_brightness": 15,
            "min_variance": 50,
            "quality_threshold": 0.60,
            "focus_weight": 0.35,
            "brightness_weight": 0.15,
            "contrast_weight": 0.20,
            "motion_weight": 0.20,
            "content_weight": 0.10
        }
    })
    
    # 🆕 OPTIMIZACIONES DE PERFORMANCE
    BATCH_SIZE_EMBEDDINGS: int = Field(default=4, description="Batch size para embeddings")
    FRAME_RESIZE_TARGET: int = Field(default=512, description="Tamaño objetivo para frames")
    ENABLE_PARALLEL_PROCESSING: bool = Field(default=True, description="Habilitar procesamiento paralelo")
    MAX_CONCURRENT_FRAMES: int = Field(default=4, description="Frames simultáneos en procesamiento")
    
    # 🆕 CONSOLIDACIÓN INTELIGENTE
    CONSOLIDATION_WEIGHTS: Dict[str, Dict[str, float]] = Field(default={
        "very_short": {  # Para videos muy cortos - priorizar score
            "mean_score": 0.50,
            "max_score": 0.35,
            "consistency": 0.10,
            "frequency": 0.05
        },
        "short": {  # Para videos cortos - balanceado
            "mean_score": 0.45,
            "max_score": 0.30,
            "consistency": 0.15,
            "frequency": 0.10
        },
        "standard": {  # Para videos normales - priorizar consistencia
            "mean_score": 0.40,
            "max_score": 0.30,
            "consistency": 0.20,
            "frequency": 0.10
        }
    })
    
    # 🆕 VERTEX AI OPTIMIZATIONS
    VERTEX_AI_BATCH_SIZE: int = Field(default=4, description="Batch size para Vertex AI")
    VERTEX_AI_TIMEOUT_SECONDS: int = Field(default=30, description="Timeout para Vertex AI")
    VERTEX_AI_RETRY_ATTEMPTS: int = Field(default=2, description="Reintentos para Vertex AI")
    
    # 🆕 MEMORY MANAGEMENT
    MAX_MEMORY_MB_PER_REQUEST: int = Field(default=200, description="Memoria máxima por request")
    CLEANUP_TEMP_FILES_IMMEDIATELY: bool = Field(default=True, description="Limpieza inmediata de archivos temp")
    
    # Video processing (original)
    MAX_VIDEO_SIZE_MB: int = Field(default=10, description="Max video size in MB")
    MAX_PROCESSING_TIME_MINUTES: int = Field(default=3, description="Max processing time - AUMENTADO")
    
    # Storage
    TEMP_STORAGE_PATH: str = Field(default="/tmp/videos", description="Temp storage for videos")
    
    # 🆕 CONFIDENCE THRESHOLDS
    MIN_CONFIDENCE_FOR_TRAINING: float = Field(default=0.85, description="Confianza mínima para agregar a entrenamiento")
    MIN_CONFIDENCE_FOR_RESPONSE: float = Field(default=0.30, description="Confianza mínima para incluir en respuesta")
    
    class Config:
        env_file = ".env"

settings = Settings()