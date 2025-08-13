# video-processing-microservice/app/services/video_processor.py
import cv2
import numpy as np
import os
import tempfile
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from fastapi import UploadFile
from .embedding_service import EmbeddingService
from .pinecone_service import PineconeService
from app.config.settings import settings

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Procesador principal de videos"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.temp_path = settings.TEMP_STORAGE_PATH
        os.makedirs(self.temp_path, exist_ok=True)
    
    async def process_video_complete(
        self,
        job_id: int,
        video_file: UploadFile,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesamiento completo de video con IA
        """
        video_path = None
        try:
            logger.info(f"🎬 Iniciando procesamiento video - Job ID: {job_id}")
            
            # 1. Guardar video temporalmente
            video_path = await self._save_video_temp(video_file, job_id)
            
            # 2. Validar video
            await self._validate_video(video_path)
            
            # 3. Extraer frames clave
            logger.info(f"🎯 Extrayendo frames - Job ID: {job_id}")
            frames = await self._extract_key_frames(video_path)
            
            # 4. Procesar cada frame con IA
            logger.info(f"🧠 Procesando {len(frames)} frames con IA - Job ID: {job_id}")
            frame_results = []
            
            for i, frame in enumerate(frames):
                # Generar embedding
                embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                
                # Buscar productos similares
                similar_products = await self.pinecone_service.search_similar(
                    embedding, top_k=5, confidence_threshold=0.7
                )
                
                frame_results.append({
                    "frame_index": i,
                    "timestamp_seconds": i * 2,  # Asumiendo frames cada 2 segundos
                    "similar_products": similar_products,
                    "confidence_scores": [p.get("score", 0) for p in similar_products]
                })
                
                logger.info(f"Frame {i+1}/{len(frames)} procesado - {len(similar_products)} productos encontrados")
            
            # 5. Consolidar resultados
            logger.info(f"🔄 Consolidando resultados - Job ID: {job_id}")
            consolidated_result = await self._consolidate_frame_results(frame_results, metadata)
            
            # 6. Agregar datos de entrenamiento a Pinecone
            if consolidated_result.get("should_add_to_training", True):
                logger.info(f"📚 Agregando a entrenamiento - Job ID: {job_id}")
                await self._add_training_data(frames, consolidated_result, metadata)
            
            logger.info(f"✅ Procesamiento completado - Job ID: {job_id}")
            
            return {
                "status": "completed",
                "job_id": job_id,
                "processing_time_seconds": 120,  # Se calcularía el tiempo real
                "frames_processed": len(frames),
                "detected_products": consolidated_result.get("detected_products", []),
                "confidence_score": consolidated_result.get("overall_confidence", 0.0),
                "ai_extracted_info": consolidated_result.get("ai_info", {}),
                "training_data_added": consolidated_result.get("should_add_to_training", True)
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando video {job_id}: {e}")
            raise Exception(f"Error procesando video: {str(e)}")
            
        finally:
            # Limpiar archivo temporal
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    logger.info(f"🗑️ Archivo temporal eliminado - Job ID: {job_id}")
                except:
                    pass
    
    async def _save_video_temp(self, video_file: UploadFile, job_id: int) -> str:
        """Guardar video temporalmente"""
        file_extension = video_file.filename.split('.')[-1] if video_file.filename else 'mp4'
        temp_filename = f"job_{job_id}_{int(asyncio.get_event_loop().time())}.{file_extension}"
        temp_path = os.path.join(self.temp_path, temp_filename)
        
        with open(temp_path, "wb") as temp_file:
            content = await video_file.read()
            temp_file.write(content)
        
        logger.info(f"💾 Video guardado temporalmente: {temp_path}")
        return temp_path
    
    async def _validate_video(self, video_path: str):
        """Validar que el video sea procesable"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("No se puede abrir el archivo de video")
        
        # Verificar duración
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        if duration > settings.MAX_PROCESSING_TIME_MINUTES * 60:
            raise Exception(f"Video muy largo: {duration:.1f}s. Máximo permitido: {settings.MAX_PROCESSING_TIME_MINUTES * 60}s")
        
        logger.info(f"✅ Video validado - Duración: {duration:.1f}s, FPS: {fps}")
    
    # REEMPLAZAR EN video_processor.py
    async def _extract_key_frames(self, video_path: str) -> List[np.ndarray]:
        """Extracción inteligente de frames para reconocimiento de productos"""
        
        # 1. Configuración adaptativa
        config = self._get_adaptive_config(video_path)
        
        # 2. Análisis multi-criterio
        candidates = await self._analyze_all_frames(video_path, config)
        
        # 3. Selección óptima
        selected_indices = self._select_optimal_frames(candidates, config["max_frames"])
        
        # 4. Extracción y optimización
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        for idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Optimización específica para reconocimiento
                optimized = await self._optimize_for_product_recognition(frame)
                frames.append(optimized)
        
        cap.release()
        
        logger.info(f"🎯 Extraídos {len(frames)} frames óptimos de {selected_indices[-1]} totales")
        return frames
    
    async def _consolidate_frame_results(
        self,
        frame_results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Consolidar resultados de todos los frames"""
        
        # Agregar lógica de consolidación inteligente
        all_products = []
        confidence_scores = []
        
        for frame_result in frame_results:
            all_products.extend(frame_result.get("similar_products", []))
            confidence_scores.extend(frame_result.get("confidence_scores", []))
        
        # Encontrar productos más frecuentes
        product_frequency = {}
        for product in all_products:
            model_name = product.get("model_name", "unknown")
            brand = product.get("brand", "unknown")
            key = f"{brand}_{model_name}"
            
            if key not in product_frequency:
                product_frequency[key] = {
                    "product": product,
                    "count": 0,
                    "total_confidence": 0.0
                }
            
            product_frequency[key]["count"] += 1
            product_frequency[key]["total_confidence"] += product.get("score", 0)
        
        # Ordenar por frecuencia y confianza
        sorted_products = sorted(
            product_frequency.items(),
            key=lambda x: (x[1]["count"], x[1]["total_confidence"]),
            reverse=True
        )
        
        detected_products = []
        for key, data in sorted_products[:3]:  # Top 3 productos
            product = data["product"]
            avg_confidence = data["total_confidence"] / data["count"]
            
            detected_products.append({
                "brand": product.get("brand"),
                "model_name": product.get("model_name"),
                "color": product.get("color"),
                "confidence": round(avg_confidence, 3),
                "detection_frequency": data["count"],
                "recommended_reference_code": self._generate_reference_code(
                    product.get("brand"), 
                    product.get("model_name")
                )
            })
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            "detected_products": detected_products,
            "overall_confidence": round(overall_confidence, 3),
            "total_frames_analyzed": len(frame_results),
            "ai_info": {
                "processing_method": "multi_frame_analysis",
                "consolidation_strategy": "frequency_weighted",
                "confidence_threshold": 0.7
            },
            "should_add_to_training": overall_confidence > 0.8  # Solo agregar si confianza alta
        }
    
    def _generate_reference_code(self, brand: str, model: str) -> str:
        """Generar código de referencia único"""
        import uuid
        
        brand_code = (brand or "UNK")[:3].upper()
        model_code = (model or "MDL")[:4].upper()
        unique_suffix = str(uuid.uuid4())[:6].upper()
        
        return f"{brand_code}-{model_code}-{unique_suffix}"
    
    async def _add_training_data(
        self,
        frames: List[np.ndarray],
        consolidated_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Agregar frames como datos de entrenamiento a Pinecone"""
        
        detected_products = consolidated_result.get("detected_products", [])
        if not detected_products:
            logger.warning("No hay productos detectados para entrenamiento")
            return
        
        # Usar el producto con mayor confianza
        best_product = detected_products[0]
        
        for i, frame in enumerate(frames):
            try:
                # Generar embedding del frame
                embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                
                # Crear metadata para Pinecone
                pinecone_metadata = {
                    "brand": best_product.get("brand"),
                    "model_name": best_product.get("model_name"),
                    "color": best_product.get("color"),
                    "confidence": best_product.get("confidence"),
                    "source": "admin_video_training",
                    "warehouse_id": metadata.get("warehouse_id"),
                    "admin_id": metadata.get("admin_id"),
                    "frame_index": i,
                    "training_session": metadata.get("job_db_id")
                }
               
               # Vector ID único
                vector_id = f"training_{metadata.get('job_db_id')}_{i}_{int(asyncio.get_event_loop().time())}"
               
               # Agregar a Pinecone
                await self.pinecone_service.upsert_vector(vector_id, embedding, pinecone_metadata)
               
                logger.info(f"📚 Frame {i+1} agregado a entrenamiento - Vector ID: {vector_id}")
               
            except Exception as e:
                logger.error(f"❌ Error agregando frame {i} a entrenamiento: {e}")
                continue
       
        logger.info(f"✅ {len(frames)} frames agregados como datos de entrenamiento")