# video-processing-microservice/app/services/video_processor_optimized.py
import cv2
import numpy as np
import os
import tempfile
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from fastapi import UploadFile
import time
from concurrent.futures import ThreadPoolExecutor
from .embedding_service import EmbeddingService
from .pinecone_service import PineconeService
from app.config.settings import settings

logger = logging.getLogger(__name__)

class VideoProcessorOptimized:
    """🚀 Procesador de video ultra-optimizado para videos cortos (3-5 segundos)"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.temp_path = settings.TEMP_STORAGE_PATH
        os.makedirs(self.temp_path, exist_ok=True)
        
        # Thread pool para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_FRAMES)
        
        # Cache para frames anteriores (detección de movimiento)
        self._previous_frame_cache = {}
        
        logger.info("🚀 VideoProcessorOptimized inicializado con configuración adaptativa")
    
    async def process_video_complete(
        self,
        job_id: int,
        video_file: UploadFile,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """🎬 Procesamiento completo optimizado para videos cortos"""
        
        start_time = time.time()
        video_path = None
        
        try:
            logger.info(f"🎬 INICIANDO PROCESAMIENTO OPTIMIZADO - Job ID: {job_id}")
            
            # 1. Guardar y validar video
            video_path = await self._save_video_temp(video_file, job_id)
            duration = await self._validate_and_get_duration(video_path)
            
            # 2. Configuración adaptativa según duración
            config = self._get_adaptive_config(duration)
            logger.info(f"⚙️ Estrategia seleccionada: {config['strategy']} para {duration:.1f}s")
            
            # 3. Extracción inteligente de frames
            logger.info(f"🎯 Extrayendo frames con estrategia {config['strategy']}")
            frames = await self._extract_key_frames_optimized(video_path, config)
            
            # 4. Procesamiento por lotes de embeddings
            logger.info(f"🧠 Procesando {len(frames)} frames en lotes optimizados")
            frame_results = await self._process_frames_batch_optimized(frames, config)
            
            # 5. Consolidación estadística inteligente
            logger.info(f"📊 Consolidando resultados con análisis estadístico")
            consolidated_result = await self._consolidate_results_advanced(
                frame_results, metadata, config
            )
            
            # 6. Entrenamiento inteligente (solo si alta confianza)
            training_added = False
            if consolidated_result.get("should_add_to_training", False):
                logger.info(f"📚 Agregando a entrenamiento - Confianza: {consolidated_result['overall_confidence']:.3f}")
                training_added = await self._add_training_data_selective(
                    frames, consolidated_result, metadata
                )
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "completed",
                "job_id": job_id,
                "processing_time_seconds": round(processing_time, 2),
                "frames_processed": len(frames),
                "video_duration_seconds": round(duration, 1),
                "strategy_used": config['strategy'],
                "detected_products": consolidated_result.get("detected_products", []),
                "confidence_score": consolidated_result.get("overall_confidence", 0.0),
                "ai_extracted_info": consolidated_result.get("ai_info", {}),
                "training_data_added": training_added,
                "performance_metrics": {
                    "frames_per_second_processed": round(len(frames) / processing_time, 2),
                    "strategy": config['strategy'],
                    "optimization_level": "ultra_optimized"
                }
            }
            
            logger.info(f"✅ PROCESAMIENTO COMPLETADO - Job {job_id} - {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando video {job_id}: {e}")
            raise Exception(f"Error en procesamiento optimizado: {str(e)}")
            
        finally:
            # Limpieza inmediata
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    logger.debug(f"🗑️ Cleanup completado - Job ID: {job_id}")
                except:
                    pass
            
            # Limpiar cache de frames anteriores
            self._previous_frame_cache.pop(job_id, None)
    
    def _get_adaptive_config(self, duration: float) -> Dict[str, Any]:
        """⚙️ Configuración adaptativa basada en duración del video"""
        
        if duration <= settings.VERY_SHORT_VIDEO_THRESHOLD_SECONDS:
            # Videos muy cortos (≤3s) - Máxima densidad
            return {
                'strategy': 'ultra_dense_sampling',
                'interval_seconds': settings.DENSE_SAMPLING_INTERVAL_SECONDS,
                'max_frames': settings.MAX_FRAMES_VERY_SHORT_VIDEO,
                'quality_config': settings.QUALITY_FILTERS['very_short'],
                'consolidation_weights': settings.CONSOLIDATION_WEIGHTS['very_short'],
                'priority': 'maximize_coverage'
            }
        elif duration <= settings.SHORT_VIDEO_THRESHOLD_SECONDS:
            # Videos cortos (3-6s) - Alta densidad
            return {
                'strategy': 'balanced_dense_sampling',
                'interval_seconds': settings.BALANCED_SAMPLING_INTERVAL_SECONDS,
                'max_frames': settings.MAX_FRAMES_SHORT_VIDEO,
                'quality_config': settings.QUALITY_FILTERS['short'],
                'consolidation_weights': settings.CONSOLIDATION_WEIGHTS['short'],
                'priority': 'balance_quality_coverage'
            }
        else:
            # Videos normales (>6s) - Calidad prioritaria
            return {
                'strategy': 'quality_focused_sampling',
                'interval_seconds': settings.STANDARD_SAMPLING_INTERVAL_SECONDS,
                'max_frames': settings.MAX_FRAMES_TO_EXTRACT,
                'quality_config': settings.QUALITY_FILTERS['standard'],
                'consolidation_weights': settings.CONSOLIDATION_WEIGHTS['standard'],
                'priority': 'maximize_quality'
            }
    
    async def _validate_and_get_duration(self, video_path: str) -> float:
        """✅ Validación rápida y obtención de duración"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("No se puede abrir el archivo de video")
        
        # Obtener métricas básicas
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Validaciones
        if duration > settings.MAX_PROCESSING_TIME_MINUTES * 60:
            raise Exception(f"Video muy largo: {duration:.1f}s. Máximo: {settings.MAX_PROCESSING_TIME_MINUTES * 60}s")
        
        if duration < 0.5:
            raise Exception("Video muy corto: mínimo 0.5 segundos")
        
        logger.info(f"✅ Video validado - Duración: {duration:.1f}s, FPS: {fps:.1f}")
        return duration
    
    async def _extract_key_frames_optimized(
        self,
        video_path: str,
        config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """🎯 Extracción ultra-optimizada de frames con filtros inteligentes"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        interval_frames = max(1, int(fps * config['interval_seconds']))
        quality_config = config['quality_config']
        
        frame_candidates = []
        frame_id = 0
        
        logger.debug(f"📊 Analizando video - Total frames: {total_frames}, Intervalo: {interval_frames}")
        
        # FASE 1: Recopilar candidatos con análisis de calidad
        while cap.isOpened() and len(frame_candidates) < config['max_frames'] * 2:  # 2x para tener opciones
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % interval_frames == 0:
                # Análisis rápido de calidad
                quality_metrics = await self._compute_quality_metrics_fast(
                    frame, quality_config, frame_id
                )
                
                if quality_metrics['total_score'] > quality_config['quality_threshold']:
                    frame_candidates.append({
                        'frame': frame.copy(),
                        'frame_id': frame_id,
                        'timestamp': frame_id / fps,
                        'quality_score': quality_metrics['total_score'],
                        'metrics': quality_metrics
                    })
            
            frame_id += 1
        
        cap.release()
        
        logger.debug(f"📊 Candidatos encontrados: {len(frame_candidates)}")
        
        # FASE 2: Selección inteligente según estrategia
        if config['strategy'] == 'ultra_dense_sampling':
            # Para videos muy cortos: tomar los mejores frames disponibles
            selected_candidates = sorted(
                frame_candidates,
                key=lambda x: x['quality_score'],
                reverse=True
            )[:config['max_frames']]
            
        elif config['strategy'] == 'balanced_dense_sampling':
            # Balance entre calidad y distribución temporal
            selected_candidates = await self._balanced_frame_selection(
                frame_candidates, config['max_frames']
            )
            
        else:  # quality_focused_sampling
            # Priorizar máxima calidad
            high_quality = [
                c for c in frame_candidates
                if c['quality_score'] > quality_config['quality_threshold'] * 1.2
            ]
            
            selected_candidates = sorted(
                high_quality,
                key=lambda x: x['quality_score'],
                reverse=True
            )[:config['max_frames']]
        
        # FASE 3: Optimización paralela de frames seleccionados
        if settings.ENABLE_PARALLEL_PROCESSING:
            optimized_frames = await self._optimize_frames_parallel([c['frame'] for c in selected_candidates])
        else:
            optimized_frames = []
            for candidate in selected_candidates:
                optimized = await self._optimize_frame_for_recognition(candidate['frame'])
                optimized_frames.append(optimized)
        
        logger.info(f"🎯 Frames finales extraídos: {len(optimized_frames)} (estrategia: {config['strategy']})")
        return optimized_frames
    
    async def _compute_quality_metrics_fast(
        self,
        frame: np.ndarray,
        quality_config: Dict[str, Any],
        frame_id: int
    ) -> Dict[str, float]:
        """⚡ Métricas de calidad ultra-rápidas"""
        
        # Redimensionar para análisis rápido (4x más rápido)
        small_frame = cv2.resize(frame, (160, 120))  # Más pequeño para más velocidad
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Brillo (muy rápido)
        brightness = np.mean(gray_small)
        brightness_score = min(1.0, brightness / 100.0)
        
        # 2. Enfoque usando gradientes Sobel (más rápido que Laplacian)
        grad_x = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
        focus_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        focus_score = min(1.0, focus_magnitude / 15.0)
        
        # 3. Contraste (muy rápido)
        contrast = gray_small.std()
        contrast_score = min(1.0, contrast / 40.0)
        
        # 4. Detección de movimiento/blur
        cache_key = f"prev_frame"
        if cache_key in self._previous_frame_cache:
            diff = cv2.absdiff(gray_small, self._previous_frame_cache[cache_key])
            motion_blur = np.mean(diff)
            motion_score = max(0.0, 1.0 - (motion_blur / 30.0))  # Menos movimiento = mejor
        else:
            motion_score = 1.0
        
        self._previous_frame_cache[cache_key] = gray_small.copy()
        
        # 5. Puntuación total con pesos configurables
        weights = quality_config
        total_score = (
            brightness_score * weights.get('brightness_weight', 0.15) +
            focus_score * weights.get('focus_weight', 0.40) +
            contrast_score * weights.get('contrast_weight', 0.25) +
            motion_score * weights.get('motion_weight', 0.20)
        )
        
        return {
            'brightness': brightness_score,
            'focus': focus_score,
            'contrast': contrast_score,
            'motion': motion_score,
            'total_score': total_score
        }
    
    async def _balanced_frame_selection(
        self,
        candidates: List[Dict[str, Any]],
        max_frames: int
    ) -> List[Dict[str, Any]]:
        """⚖️ Selección balanceada entre calidad y distribución temporal"""
        
        if len(candidates) <= max_frames:
            return candidates
        
        # Dividir en segmentos temporales
        duration = max(c['timestamp'] for c in candidates)
        num_segments = min(max_frames, max(2, len(candidates) // 3))
        segment_duration = duration / num_segments
        
        selected = []
        
        for segment in range(num_segments):
            start_time = segment * segment_duration
            end_time = (segment + 1) * segment_duration
            
            segment_candidates = [
                c for c in candidates
                if start_time <= c['timestamp'] < end_time
            ]
            
            if segment_candidates:
                best_in_segment = max(segment_candidates, key=lambda x: x['quality_score'])
                selected.append(best_in_segment)
        
        # Completar con los mejores restantes
        remaining_slots = max_frames - len(selected)
        if remaining_slots > 0:
            used_timestamps = {c['timestamp'] for c in selected}
            remaining = [c for c in candidates if c['timestamp'] not in used_timestamps]
            
            best_remaining = sorted(remaining, key=lambda x: x['quality_score'], reverse=True)[:remaining_slots]
            selected.extend(best_remaining)
        
        return selected[:max_frames]
    
    async def _optimize_frames_parallel(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """🚀 Optimización paralela de frames"""
        
        if not settings.ENABLE_PARALLEL_PROCESSING or len(frames) <= 2:
            # Procesamiento secuencial para pocos frames
            optimized = []
            for frame in frames:
                opt_frame = await self._optimize_frame_for_recognition(frame)
                optimized.append(opt_frame)
            return optimized
        
        # Procesamiento paralelo
        loop = asyncio.get_event_loop()
        
        tasks = []
        for frame in frames:
            task = loop.run_in_executor(
                self.executor,
                self._optimize_frame_sync,
                frame
            )
            tasks.append(task)
        
        optimized_frames = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar excepciones
        valid_frames = [f for f in optimized_frames if not isinstance(f, Exception)]
        
        if len(valid_frames) != len(frames):
            logger.warning(f"⚠️ Algunos frames fallaron en optimización: {len(valid_frames)}/{len(frames)}")
        
        return valid_frames
    
    def _optimize_frame_sync(self, frame: np.ndarray) -> np.ndarray:
        """🔧 Optimización síncrona de frame individual"""
        try:
            return asyncio.run(self._optimize_frame_for_recognition(frame))
        except Exception as e:
            logger.error(f"Error optimizando frame: {e}")
            return frame  # Devolver original si falla
    
    async def _optimize_frame_for_recognition(self, frame: np.ndarray) -> np.ndarray:
        """🎨 Optimización específica para reconocimiento de productos"""
        
        # 1. Redimensionar inteligente manteniendo aspect ratio
        height, width = frame.shape[:2]
        target_size = settings.FRAME_RESIZE_TARGET
        
        if max(height, width) > target_size:
            if width > height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Mejora de contraste adaptativo CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        l = clahe.apply(l)
        
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        # 3. Reducción de ruido bilateral (preserva bordes)
        frame = cv2.bilateralFilter(frame, 7, 50, 50)
        
        # 4. Ajuste final de brillo y contraste
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=8)
        
        return frame
    
    async def _process_frames_batch_optimized(
        self,
        frames: List[np.ndarray],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """🔄 Procesamiento por lotes de embeddings optimizado"""
        
        if not frames:
            return []
        
        batch_size = settings.BATCH_SIZE_EMBEDDINGS
        frame_results = []
        
        # Procesar en lotes para optimizar llamadas a Vertex AI
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            logger.debug(f"🔄 Procesando lote {i//batch_size + 1}/{(len(frames)-1)//batch_size + 1}")
            
            # Generar embeddings del lote
            batch_embeddings = await self._get_batch_embeddings_optimized(batch_frames)
            
            # Buscar productos similares para cada embedding
            for j, embedding in enumerate(batch_embeddings):
                frame_index = i + j
                
                # Búsqueda en Pinecone
                similar_products = await self.pinecone_service.search_similar(
                    embedding,
                    top_k=5,
                    confidence_threshold=settings.MIN_CONFIDENCE_FOR_RESPONSE
                )
                
                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index * config['interval_seconds'],
                    "similar_products": similar_products,
                    "confidence_scores": [p.get("score", 0) for p in similar_products],
                    "batch_id": i//batch_size
                })
        
        logger.info(f"🔄 Procesados {len(frame_results)} frames en {(len(frames)-1)//batch_size + 1} lotes")
        return frame_results
    
    async def _get_batch_embeddings_optimized(self, frames: List[np.ndarray]) -> List[List[float]]:
        """⚡ Generación optimizada de embeddings por lotes"""
        
        embeddings = []
        
        try:
            # Procesar frames secuencialmente (Vertex AI no soporta batch nativo)
            for frame in frames:
                embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                embeddings.append(embedding)
                
                # Pequeña pausa para evitar rate limiting
                if len(frames) > 1:
                    await asyncio.sleep(0.1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error en batch embeddings: {e}")
            # Fallback: procesar uno por uno con mayor tolerancia a errores
            embeddings = []
            for i, frame in enumerate(frames):
                try:
                    embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                    embeddings.append(embedding)
                except Exception as frame_error:
                    logger.warning(f"⚠️ Error en frame {i}: {frame_error}")
                    continue
            
            return embeddings
    
    async def _consolidate_results_advanced(
        self,
        frame_results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """📊 Consolidación avanzada con análisis estadístico completo"""
        
        if not frame_results:
            return {
                "detected_products": [],
                "overall_confidence": 0.0,
                "frames_analyzed": 0,
                "ai_info": {"error": "No frames processed"}
            }
        
        # Agrupar detecciones por modelo
        model_stats = {}
        
        for frame_result in frame_results:
            for product in frame_result.get("similar_products", []):
                model_key = f"{product.get('brand', 'Unknown')}_{product.get('model_name', 'Unknown')}"
                
                if model_key not in model_stats:
                    model_stats[model_key] = {
                        'product': product,
                        'scores': [],
                        'frame_indices': [],
                        'timestamps': []
                    }
                
                model_stats[model_key]['scores'].append(product.get('score', 0))
                model_stats[model_key]['frame_indices'].append(frame_result['frame_index'])
                model_stats[model_key]['timestamps'].append(frame_result['timestamp_seconds'])
        
        # Análisis estadístico avanzado por modelo
        weights = config['consolidation_weights']
        
        for model_key, stats in model_stats.items():
            scores = np.array(stats['scores'])
            
            # Métricas estadísticas
            stats['mean_score'] = np.mean(scores)
            stats['max_score'] = np.max(scores)
            stats['min_score'] = np.min(scores)
            stats['std_score'] = np.std(scores)
            stats['median_score'] = np.median(scores)
            stats['detection_frequency'] = len(scores)
            stats['temporal_spread'] = max(stats['timestamps']) - min(stats['timestamps'])
            
            # Consistencia (menor desviación = más consistente)
            stats['consistency_score'] = 1.0 / (1.0 + stats['std_score'])
            
            # Frecuencia relativa
            stats['frequency_ratio'] = len(scores) / len(frame_results)
            
            # Score final ponderado con pesos configurables
            stats['final_score'] = (
                stats['mean_score'] * weights['mean_score'] +
                stats['max_score'] * weights['max_score'] +
                stats['consistency_score'] * weights['consistency'] +
                stats['frequency_ratio'] * weights['frequency']
            )
        
        # Ordenar por score final
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        # Generar productos detectados
        detected_products = []
        for i, (model_key, stats) in enumerate(sorted_models[:3]):  # Top 3
            product = stats['product']
            
            detected_products.append({
                'rank': i + 1,
                'brand': product.get('brand', 'Unknown'),
                'model_name': product.get('model_name', 'Unknown'),
                'color': product.get('color', 'Unknown'),
                'confidence': round(stats['final_score'] * 100, 2),
                'mean_similarity': round(stats['mean_score'], 3),
                'max_similarity': round(stats['max_score'], 3),
                'consistency_score': round(stats['consistency_score'], 3),
                'detection_frequency': stats['detection_frequency'],
                'frequency_percentage': round(stats['frequency_ratio'] * 100, 1),
                'temporal_spread_seconds': round(stats['temporal_spread'], 1),
                'frames_detected': stats['frame_indices'],
                'statistical_metrics': {
                    'median_score': round(stats['median_score'], 3),
                    'std_deviation': round(stats['std_score'], 3),
                    'score_range': round(stats['max_score'] - stats['min_score'], 3)
                }
            })
        
        # Confianza general
        overall_confidence = sorted_models[0][1]['final_score'] if sorted_models else 0.0
        
        # Información AI detallada
        ai_info = {
            "processing_method": "ultra_optimized_multi_frame",
            "strategy_used": config['strategy'],
            "consolidation_method": "weighted_statistical_analysis",
            "quality_filters": list(config['quality_config'].keys()),
            "batch_processing": True,
            "parallel_optimization": settings.ENABLE_PARALLEL_PROCESSING,
            "models_analyzed": len(model_stats),
            "statistical_confidence": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.5 else "low"
        }
        
        return {
            "detected_products": detected_products,
            "overall_confidence": round(overall_confidence, 3),
            "frames_analyzed": len(frame_results),
            "models_considered": len(model_stats),
            "ai_info": ai_info,
            "should_add_to_training": overall_confidence >= settings.MIN_CONFIDENCE_FOR_TRAINING
        }
    
    async def _add_training_data_selective(
        self,
        frames: List[np.ndarray],
        consolidated_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """📚 Agregar datos de entrenamiento de forma selectiva"""
        
        detected_products = consolidated_result.get("detected_products", [])
        if not detected_products:
            logger.warning("⚠️ No hay productos detectados para entrenamiento")
            return False
        
        # Usar solo el producto con mayor confianza
        best_product = detected_products[0]
        
        if best_product['confidence'] < settings.MIN_CONFIDENCE_FOR_TRAINING * 100:
            logger.info(f"⚠️ Confianza insuficiente para entrenamiento: {best_product['confidence']}%")
            return False
        
        # Seleccionar solo los mejores frames (top 50%)
        num_frames_for_training = max(1, len(frames) // 2)
        selected_frames = frames[:num_frames_for_training]
        
        training_success_count = 0
        
        for i, frame in enumerate(selected_frames):
            try:
                # Generar embedding para entrenamiento
                embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                
                # Metadata enriquecida para Pinecone
                training_metadata = {
                    "brand": best_product.get("brand"),
                    "model_name": best_product.get("model_name"),
                    "color": best_product.get("color"),
                    "confidence": best_product.get("confidence") / 100.0,
                    "source": "optimized_video_training",
                    "warehouse_id": metadata.get("warehouse_id"),
                    "admin_id": metadata.get("admin_id"),
                    "frame_index": i,
                    "training_session": metadata.get("job_db_id"),
                    "optimization_level": "ultra_optimized",
                    "statistical_confidence": consolidated_result.get("ai_info", {}).get("statistical_confidence"),
                    "training_timestamp": int(time.time())
                }
                
                # Vector ID único con timestamp
                vector_id = f"opt_train_{metadata.get('job_db_id', 'unknown')}_{i}_{int(time.time() * 1000)}"
                
                # Agregar a Pinecone
                success = await self.pinecone_service.upsert_vector(
                    vector_id, embedding, training_metadata
                )
                
                if success:
                    training_success_count += 1
                    logger.debug(f"📚 Frame {i+1} agregado a entrenamiento - Vector: {vector_id}")
                
                # Pausa pequeña para evitar rate limiting
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"❌ Error agregando frame {i} a entrenamiento: {e}")
                continue
        
        success_rate = training_success_count / len(selected_frames)
        logger.info(f"✅ Entrenamiento completado: {training_success_count}/{len(selected_frames)} frames ({success_rate:.1%})")
        
        return success_rate > 0.5  # Considerar exitoso si >50% de frames se agregaron
    
    async def _save_video_temp(self, video_file: UploadFile, job_id: int) -> str:
        """💾 Guardar video temporalmente con nombre optimizado"""
        
        file_extension = 'mp4'
        if video_file.filename:
            file_extension = video_file.filename.split('.')[-1].lower()
        
        timestamp = int(time.time() * 1000)  # Más precisión
        temp_filename = f"opt_job_{job_id}_{timestamp}.{file_extension}"
        temp_path = os.path.join(self.temp_path, temp_filename)
        
        # Escribir archivo de forma eficiente
        content = await video_file.read()
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Verificar tamaño
        file_size_mb = len(content) / (1024 * 1024)
        logger.debug(f"💾 Video guardado: {temp_path} ({file_size_mb:.1f}MB)")
        
        return temp_path

# Crear instancia global optimizada
video_processor_optimized = VideoProcessorOptimized()