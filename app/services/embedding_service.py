# video-processing-microservice/app/services/embedding_service.py
import logging
import tempfile
import os
import numpy as np
from typing import List
import asyncio
from app.config.settings import settings

# Importaciones de Google Cloud
try:
    import vertexai
    from vertexai.vision_models import Image as VertexImage, MultiModalEmbeddingModel
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    vertexai = None
    VertexImage = None
    MultiModalEmbeddingModel = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Servicio de embeddings optimizado para microservicio"""
    
    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.location = settings.GOOGLE_CLOUD_LOCATION
        self.model = None
        self.embedding_count = 0
        
        if not GOOGLE_AVAILABLE:
            logger.error("❌ Google Vertex AI no disponible")
            return
        
        # Inicializar modelo inmediatamente (microservicio dedicado)
        asyncio.create_task(self._initialize_model())
    
    async def _initialize_model(self):
        """Inicializar modelo Google Multimodal"""
        try:
            logger.info("🔄 Inicializando Google Multimodal...")
            
            vertexai.init(project=self.project_id, location=self.location)
            self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            
            logger.info("✅ Google Multimodal inicializado en microservicio")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando modelo: {e}")
            self.model = None
    
    async def get_image_embedding_from_array(self, image_array: np.ndarray) -> List[float]:
        """Generar embedding desde array de imagen (frame de video)"""
        
        # Esperar inicialización si es necesario
        max_wait = 30
        waited = 0
        while self.model is None and waited < max_wait:
            await asyncio.sleep(1)
            waited += 1
        
        if self.model is None:
            raise Exception("Modelo de embeddings no disponible")
        
        try:
            # Convertir array numpy a imagen temporal
            import cv2
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                # Convertir BGR a RGB si es necesario
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                
                # Guardar como imagen temporal
                cv2.imwrite(temp_file.name, image_rgb)
                temp_path = temp_file.name
            
            try:
                # Cargar imagen para Vertex AI
                vertex_image = VertexImage.load_from_file(temp_path)
                
                # Generar embedding
                embeddings_response = self.model.get_embeddings(
                    image=vertex_image,
                    dimension=1408
                )
                
                embedding = embeddings_response.image_embedding
                
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                self.embedding_count += 1
                
                logger.debug(f"🔥 Embedding generado - Dimensiones: {len(embedding_list)}")
                return embedding_list
                
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            raise Exception(f"Error en embedding: {str(e)}")
    
    async def health_check(self) -> bool:
        """Health check del servicio"""
        return self.model is not None