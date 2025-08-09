# video-processing-microservice/app/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Video Processing Microservice"
    VERSION: str = "1.0.0"
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
    
    # Video processing
    MAX_VIDEO_SIZE_MB: int = Field(default=100, description="Max video size in MB")
    MAX_PROCESSING_TIME_MINUTES: int = Field(default=10, description="Max processing time")
    MAX_FRAMES_TO_EXTRACT: int = Field(default=10, description="Max frames to extract")
    
    # Storage
    TEMP_STORAGE_PATH: str = Field(default="/tmp/videos", description="Temp storage for videos")
    
    class Config:
        env_file = ".env"

settings = Settings()