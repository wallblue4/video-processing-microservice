# video-processing-microservice/app/services/pinecone_service.py
import pinecone
from typing import List, Dict, Optional, Any
import logging
import asyncio
from app.config.settings import settings

logger = logging.getLogger(__name__)

class PineconeService:
    """Servicio Pinecone optimizado para microservicio"""
    
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.pc = None
        self.index = None
        
        if self.api_key:
            try:
                self.pc = pinecone.Pinecone(api_key=self.api_key)
                self.index = self.pc.Index(self.index_name)
                logger.info(f"✅ Pinecone inicializado - Índice: {self.index_name}")
            except Exception as e:
                logger.error(f"❌ Error inicializando Pinecone: {e}")
        else:
            logger.warning("⚠️ PINECONE_API_KEY no configurada")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        confidence_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Buscar productos similares"""
        
        if not self.index:
            logger.error("❌ Pinecone no inicializado")
            return []
        
        try:
            # Ejecutar búsqueda en thread pool para evitar bloqueo
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            )
            
            # Filtrar por umbral de confianza
            results = []
            for match in response.matches:
                if match.score >= confidence_threshold:
                    result = {
                        "id": match.id,
                        "score": match.score,
                        "confidence": round(match.score * 100, 1)
                    }
                    
                    # Agregar metadata si existe
                    if match.metadata:
                        result.update(match.metadata)
                    
                    results.append(result)
            
            logger.info(f"🔍 Encontrados {len(results)} productos similares (umbral: {confidence_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda Pinecone: {e}")
            return []
    
    async def upsert_vector(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Agregar vector de entrenamiento"""
        
        if not self.index:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            )
            
            logger.debug(f"📚 Vector {vector_id} agregado a entrenamiento")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error agregando vector {vector_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Health check de Pinecone"""
        if not self.index:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(None, self.index.describe_index_stats)
            return stats is not None
        except:
            return False