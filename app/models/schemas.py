# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class VideoProcessingRequest(BaseModel):
    job_id: int = Field(..., description="ID del job")
    callback_url: str = Field(..., description="URL para callback")
    metadata: Dict[str, Any] = Field(..., description="Metadata del job")

class VideoProcessingResponse(BaseModel):
    status: str
    job_id: int
    message: str
    estimated_time_minutes: str

class ProcessingResult(BaseModel):
    status: str
    job_id: int
    processing_time_seconds: float
    frames_processed: int
    detected_products: List[Dict[str, Any]]
    confidence_score: float
    ai_extracted_info: Dict[str, Any]
    training_data_added: bool

class HealthResponse(BaseModel):
    service: str
    version: str
    status: str
    components: Dict[str, str]
    config: Dict[str, Any]