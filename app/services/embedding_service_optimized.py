# video-processing-microservice/app/services/embedding_service_optimized.py
import logging
import tempfile
import os
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time
from app.config.settings import settings

# Importaciones de Google Cloud con timeout
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

class EmbeddingServiceOptimized:
    """🚀 Servicio de embeddings ultra-optimizado con batch processing y retry logic"""
    
    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.location = settings.GOOGLE_CLOUD_LOCATION
        self.model = None
        self.embedding_count = 0
        self.error_count = 0
        self.last_request_time = 0
        
        # Rate limiting
        self.min_request_interval = 0.1  # 100ms entre requests
        
        if not GOOGLE_AVAILABLE:
            logger.error("❌ Google Vertex AI no disponible")
            return
        
        # Inicializar modelo de forma asíncrona
        asyncio.create_task(self._initialize_model_with_retry())
    
    async def _initialize_model_with_retry(self):
        """🔄 Inicializar modelo con reintentos"""
        
        max_attempts = settings.VERTEX_AI_RETRY_ATTEMPTS
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"🔄 Inicializando Google Multimodal - Intento {attempt + 1}/{max_attempts}")
                
                if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                    logger.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS no configurado")
                
                vertexai.init(project=self.project_id, location=self.location)
                self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
                
                # Test de conectividad
                await self._test_model_connectivity()
                
                logger.info("✅ Google Multimodal inicializado y probado exitosamente")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Intento {attempt + 1} fallido: {e}")
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2  # Backoff exponential
                    logger.info(f"⏳ Esperando {wait_time}s antes del siguiente intento...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("❌ Falló inicialización después de todos los intentos")
                    self.model = None
    
    async def _test_model_connectivity(self):
        """🧪 Probar conectividad del modelo"""
        
        # Crear imagen de prueba pequeña
        test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            import cv2
            cv2.imwrite(temp_file.name, test_image)
            
            try:
                vertex_image = VertexImage.load_from_file(temp_file.name)
                test_response = self.model.get_embeddings(
                    image=vertex_image,
                    dimension=1408
                )
                
                if test_response.image_embedding is None:
                    raise Exception("Respuesta vacía del modelo")
                
                logger.debug("🧪 Test de conectividad exitoso")
                
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    async def get_image_embedding_from_array_optimized(
        self,
        image_array: np.ndarray,
        retry_on_failure: bool = True
    ) -> List[float]:
        """⚡ Generar embedding con optimizaciones y retry logic"""
        
        # Rate limiting
        await self._apply_rate_limiting()
        
        # Esperar inicialización si es necesario
        max_wait = settings.VERTEX_AI_TIMEOUT_SECONDS
        waited = 0
        while self.model is None and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5
        
        if self.model is None:
            raise Exception("Modelo de embeddings no disponible después del timeout")
        
        max_attempts = settings.VERTEX_AI_RETRY_ATTEMPTS if retry_on_failure else 1
        
        for attempt in range(max_attempts):
            try:
                embedding = await self._generate_single_embedding(image_array)
                
                self.embedding_count += 1
                self.last_request_time = time.time()
                
                if attempt > 0:
                    logger.info(f"✅ Embedding exitoso en intento {attempt + 1}")
                
                return embedding
                
            except Exception as e:
                self.error_count += 1
                
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 0.5  # 0.5s, 1s, 1.5s
                    logger.warning(f"⚠️ Intento {attempt + 1} fallido: {e}. Reintentando en {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Falló embedding después de {max_attempts} intentos: {e}")
                    raise Exception(f"Error generando embedding después de reintentos: {str(e)}")
    
    async def _generate_single_embedding(self, image_array: np.ndarray) -> List[float]:
        """🎯 Generar embedding individual con timeout"""
        
        import cv2
        
        # Validar imagen
        if image_array is None or image_array.size == 0:
            raise ValueError("Array de imagen vacío o None")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            try:
                # Convertir y guardar imagen optimizada
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # Asegurar que está en RGB para Vertex AI
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                
                # Guardar con calidad optimizada para velocidad
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Balance calidad/velocidad
                success = cv2.imwrite(temp_file.name, image_rgb, encode_params)
                
                if not success:
                    raise Exception("Error guardando imagen temporal")
                
                temp_path = temp_file.name
            
            except Exception as e:
                raise Exception(f"Error preparando imagen: {str(e)}")
            
            try:
                # Cargar imagen para Vertex AI con timeout
                vertex_image = VertexImage.load_from_file(temp_path)
                
                # Generar embedding con timeout
                start_time = time.time()
                
                embeddings_response = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: self.model.get_embeddings(
                            image=vertex_image,
                            dimension=1408
                        )
                    ),
                    timeout=settings.VERTEX_AI_TIMEOUT_SECONDS
                )
                
                processing_time = time.time() - start_time
                
                embedding = embeddings_response.image_embedding
                
                if embedding is None:
                    raise Exception("Embedding vacío recibido de Vertex AI")
                
                # Convertir a lista
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                # Validar dimensiones
                if len(embedding_list) != 1408:
                    raise Exception(f"Dimensiones incorrectas: {len(embedding_list)}, esperadas: 1408")
                
                logger.debug(f"🔥 Embedding generado en {processing_time:.2f}s - Dimensiones: {len(embedding_list)}")
                return embedding_list
                
            except asyncio.TimeoutError:
                raise Exception(f"Timeout generando embedding ({settings.VERTEX_AI_TIMEOUT_SECONDS}s)")
            except Exception as e:
                raise Exception(f"Error en Vertex AI: {str(e)}")
            
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    async def _apply_rate_limiting(self):
        """⏱️ Aplicar rate limiting para evitar sobrecarga"""
        
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                await asyncio.sleep(sleep_time)
    
    async def batch_process_embeddings(
        self,
        image_arrays: List[np.ndarray],
        max_concurrent: int = None
    ) -> List[List[float]]:
        """🔄 Procesar múltiples embeddings con concurrencia controlada"""
        
        if not image_arrays:
            return []
        
        if max_concurrent is None:
            max_concurrent = min(len(image_arrays), settings.MAX_CONCURRENT_FRAMES)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(image_array: np.ndarray) -> Optional[List[float]]:
            async with semaphore:
                try:
                    return await self.get_image_embedding_from_array_optimized(image_array)
                except Exception as e:
                    logger.error(f"Error procesando embedding en batch: {e}")
                    return None
        
        logger.info(f"🔄 Procesando {len(image_arrays)} embeddings en lotes (max concurrent: {max_concurrent})")
        
        start_time = time.time()
        
        # Procesar todos los embeddings concurrentemente
        tasks = [process_single(img_array) for img_array in image_arrays]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # Filtrar resultados válidos
        valid_embeddings = []
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                logger.warning(f"⚠️ Error en batch embedding: {result}")
            elif result is not None:
                valid_embeddings.append(result)
            else:
                error_count += 1
        
        success_rate = len(valid_embeddings) / len(image_arrays)
        
        logger.info(f"✅ Batch completado: {len(valid_embeddings)}/{len(image_arrays)} exitosos "
                   f"({success_rate:.1%}) en {processing_time:.2f}s")
        
        if success_rate < 0.5:
            logger.warning(f"⚠️ Tasa de éxito baja en batch processing: {success_rate:.1%}")
        
        return valid_embeddings
    
    async def health_check(self) -> bool:
        """🏥 Health check optimizado"""
        
        if not self.model:
            return False
        
        try:
            # Test rápido con imagen pequeña
            test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            
            # Test con timeout corto
            embedding = await asyncio.wait_for(
                self.get_image_embedding_from_array_optimized(test_image, retry_on_failure=False),
                timeout=10.0
            )
            
            return len(embedding) == 1408
            
        except Exception as e:
            logger.error(f"❌ Health check falló: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Obtener estadísticas del servicio"""
        
        error_rate = self.error_count / max(self.embedding_count + self.error_count, 1)
        
        return {
            "embeddings_generated": self.embedding_count,
            "errors_count": self.error_count,
            "error_rate": round(error_rate, 3),
            "model_available": self.model is not None,
            "last_request_seconds_ago": round(time.time() - self.last_request_time, 1) if self.last_request_time else None,
            "rate_limiting_active": True,
            "min_request_interval_ms": self.min_request_interval * 1000
        }

# Instancia global optimizada
embedding_service_optimized = EmbeddingServiceOptimized()