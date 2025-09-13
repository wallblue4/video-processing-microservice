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
        """⚙️ Configuración adaptativa - ESTRATEGIA 360° ESCALABLE"""
        
        if settings.ENABLE_360_PRODUCT_MODE:
            # 🎯 MODO 360° PARA TODAS LAS DURACIONES
            logger.info(f"🔄 MODO 360° ESCALABLE ACTIVADO para video de {duration:.1f}s")
            return self._get_360_product_config_scalable(duration)
        
        # Fallback a configuraciones originales si 360° deshabilitado
        logger.info(f"⚙️ Usando configuración legacy para {duration:.1f}s")
        return self._get_legacy_config(duration)


    def _get_360_product_config_scalable(self, duration: float) -> Dict[str, Any]:
        """🎯 Configuración 360° escalable para CUALQUIER duración"""
        
        # 📊 DETERMINAR TIER BASADO EN DURACIÓN
        if duration <= 3.0:
            config_tier = "ultra_short"
            max_frames = settings.MAX_FRAMES_ULTRA_SHORT
            interval = settings.INTERVAL_ULTRA_SHORT
            training_pct = settings.TRAINING_PCT_ULTRA_SHORT
            min_vectors = max(10, int(max_frames * 0.4))
            
        elif duration <= 10.0:
            config_tier = "short"
            max_frames = settings.MAX_FRAMES_SHORT
            interval = settings.INTERVAL_SHORT
            training_pct = settings.TRAINING_PCT_SHORT
            min_vectors = max(14, int(max_frames * 0.4))
            
        elif duration <= 30.0:
            config_tier = "medium"
            max_frames = settings.MAX_FRAMES_MEDIUM
            interval = settings.INTERVAL_MEDIUM
            training_pct = settings.TRAINING_PCT_MEDIUM
            min_vectors = max(18, int(max_frames * 0.4))
            
        else:
            config_tier = "long"
            max_frames = settings.MAX_FRAMES_LONG
            interval = settings.INTERVAL_LONG
            training_pct = settings.TRAINING_PCT_LONG
            min_vectors = max(24, int(max_frames * 0.4))
        
        logger.info(f"📊 Config tier: {config_tier} | Max frames: {max_frames} | Interval: {interval}s | Training: {training_pct*100:.0f}%")
        
        return {
            'strategy': 'comprehensive_360_sampling',
            'config_tier': config_tier,
            'interval_seconds': interval,
            'max_frames': max_frames,
            'quality_config': {
                'min_brightness': 5,        # MUY permisivo para 360°
                'min_variance': 15,         # MUY permisivo
                'quality_threshold': 0.15,  # MUY permisivo
                'focus_weight': 0.30,
                'brightness_weight': 0.20,
                'contrast_weight': 0.25,
                'motion_weight': 0.25
            },
            'consolidation_weights': {
                'mean_score': 0.30,     # Menos peso al promedio individual
                'max_score': 0.20,      # Menos peso al mejor frame
                'consistency': 0.30,    # MÁS peso a consistencia entre frames
                'frequency': 0.20       # MÁS peso a frecuencia
            },
            'training_percentage': training_pct,
            'min_vectors_required': min_vectors,
            'priority': 'comprehensive_360_coverage'
        }

    def _get_legacy_config(self, duration: float) -> Dict[str, Any]:
        """⚙️ Configuraciones legacy para fallback"""
        # Las configuraciones originales que ya tenías
        if duration <= settings.VERY_SHORT_VIDEO_THRESHOLD_SECONDS:
            return {
                'strategy': 'ultra_dense_sampling',
                'interval_seconds': settings.DENSE_SAMPLING_INTERVAL_SECONDS,
                'max_frames': settings.MAX_FRAMES_VERY_SHORT_VIDEO,
                'quality_config': settings.QUALITY_FILTERS['very_short'],
                'consolidation_weights': settings.CONSOLIDATION_WEIGHTS['very_short'],
                'priority': 'maximize_coverage'
            }


    async def _extract_frames_adaptive_optimized(self, video_path: str, config: Dict[str, Any]) -> List[np.ndarray]:
        """🎯 Extracción adaptativa - CON SOPORTE 360° ESCALABLE"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if config['strategy'] == 'comprehensive_360_sampling':
            # 🆕 ESTRATEGIA 360° ESCALABLE
            config_tier = config.get('config_tier', 'unknown')
            logger.info(f"🎯 Aplicando extracción 360° escalable - Tier: {config_tier}")
            
            # Calcular frames distribuidos uniformemente
            max_frames = min(config['max_frames'], total_frames)
            
            if total_frames <= max_frames:
                # Si el video tiene pocos frames, usar todos
                frame_indices = list(range(total_frames))
            else:
                # Distribuir uniformemente
                frame_indices = [
                    int(i * total_frames / max_frames) 
                    for i in range(max_frames)
                ]
            
            # Asegurar cobertura completa (primer y último frame)
            if 0 not in frame_indices:
                frame_indices[0] = 0
            if (total_frames - 1) not in frame_indices:
                frame_indices[-1] = total_frames - 1
            
            # Remover duplicados y ordenar
            frame_indices = sorted(list(set(frame_indices)))
            
            logger.info(f"🎯 Extrayendo {len(frame_indices)} frames distribuidos uniformemente (tier: {config_tier})")
            
        else:
            # Estrategias originales
            frame_indices = await self._calculate_frame_indices_original(video_path, config)
        
        # Extraer frames seleccionados
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Optimizar frame para reconocimiento
                optimized_frame = await self._optimize_frame_for_recognition(frame)
                frames.append(optimized_frame)
        
        cap.release()
        
        logger.info(f"🎯 Frames finales extraídos: {len(frames)} (estrategia: {config['strategy']})")
        return frames
    
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
        """📊 Consolidación avanzada con análisis estadístico completo y prioridad de metadata manual"""
        
        if not frame_results:
            return {
                "detected_products": [],
                "overall_confidence": 0.0,
                "frames_analyzed": 0,
                "models_considered": 0,
                "ai_info": {"error": "No frames processed"},
                "should_add_to_training": False
            }
        
        # 🆕 PASO 1: VERIFICAR SI ES PRODUCTO NUEVO (METADATA MANUAL)
        expected_brand = metadata.get("product_brand")
        expected_model = metadata.get("product_model")
        
        if expected_brand and expected_model:
            # Es un producto nuevo: usar datos del formulario, no detección automática
            logger.info(f"🎯 PRODUCTO NUEVO DETECTADO: {expected_brand} {expected_model}")
            logger.info(f"📝 Priorizando metadata manual sobre detección automática")
            
            # Calcular confianza basada en si encontramos productos similares
            similarity_confidence = 0.0
            total_detections = 0
            
            for frame_result in frame_results:
                for product in frame_result.get("similar_products", []):
                    total_detections += 1
                    similarity_confidence += product.get("score", 0)
            
            # Promedio de similitud con productos existentes
            avg_similarity = similarity_confidence / max(total_detections, 1)
            
            # Para productos nuevos, la confianza se basa en consistencia de detección
            # Si no hay productos similares, alta confianza (producto único)
            # Si hay similares, confianza media-alta (variante de producto existente)
            manual_confidence = 0.95 if avg_similarity < 0.3 else 0.85
            
            detected_products = [{
                'rank': 1,
                'brand': expected_brand,
                'model_name': expected_model,
                'color': metadata.get('expected_colors', ['Unknown'])[0] if isinstance(metadata.get('expected_colors'), list) else 'Unknown',
                'confidence': round(manual_confidence * 100, 2),
                'mean_similarity': round(avg_similarity, 3),
                'max_similarity': round(avg_similarity, 3),
                'consistency_score': 1.0,  # Máxima consistencia por ser manual
                'detection_frequency': len(frame_results),  # Presente en todos los frames
                'frequency_percentage': 100.0,
                'temporal_spread_seconds': len(frame_results) * config.get('interval_seconds', 0.1),
                'frames_detected': list(range(len(frame_results))),
                'detection_method': 'manual_form_input',
                'similar_products_found': total_detections,
                'avg_similarity_to_existing': round(avg_similarity, 3),
                'statistical_metrics': {
                    'median_score': round(avg_similarity, 3),
                    'std_deviation': 0.0,  # Sin desviación por ser manual
                    'score_range': 0.0
                }
            }]
            
            ai_info = {
                "processing_method": "manual_metadata_priority",
                "strategy_used": config['strategy'],
                "consolidation_method": "manual_form_input_override",
                "detection_source": "admin_form_submission",
                "frames_analyzed": len(frame_results),
                "manual_override": True,
                "expected_brand": expected_brand,
                "expected_model": expected_model,
                "automatic_detections_found": total_detections,
                "avg_similarity_to_existing_products": round(avg_similarity, 3)
            }
            
            return {
                "detected_products": detected_products,
                "overall_confidence": manual_confidence,
                "frames_analyzed": len(frame_results),
                "models_considered": 1,  # Solo el modelo manual
                "ai_info": ai_info,
                "should_add_to_training": True,  # Siempre entrenar productos manuales
                "training_source": "new_product_manual_entry",
                "consolidation_type": "manual_metadata_priority"
            }
        
        # 🔄 PASO 2: CONSOLIDACIÓN AUTOMÁTICA (cuando NO hay metadata manual)
        logger.info("🤖 No hay metadata manual - usando consolidación automática")
        
        # Agrupar detecciones por modelo
        model_stats = {}
        
        for frame_result in frame_results:
            for product in frame_result.get("similar_products", []):
                brand = product.get("brand", "Unknown")
                model_name = product.get("model_name", "Unknown")
                color = product.get("color", "Unknown")
                
                # Crear key único para el modelo
                model_key = f"{brand}_{model_name}_{color}"
                
                if model_key not in model_stats:
                    model_stats[model_key] = {
                        'product': product,
                        'scores': [],
                        'frame_indices': [],
                        'detection_frequency': 0,
                        'total_score': 0,
                        'max_score': 0,
                        'min_score': 1.0
                    }
                
                # Agregar detección
                score = product.get("score", 0)
                model_stats[model_key]['scores'].append(score)
                model_stats[model_key]['frame_indices'].append(frame_result.get('frame_index', 0))
                model_stats[model_key]['detection_frequency'] += 1
                model_stats[model_key]['total_score'] += score
                model_stats[model_key]['max_score'] = max(model_stats[model_key]['max_score'], score)
                model_stats[model_key]['min_score'] = min(model_stats[model_key]['min_score'], score)
        
        if not model_stats:
            return {
                "detected_products": [],
                "overall_confidence": 0.0,
                "frames_analyzed": len(frame_results),
                "models_considered": 0,
                "ai_info": {"error": "No products detected in any frame"},
                "should_add_to_training": False
            }
        
        # Calcular métricas estadísticas avanzadas
        total_frames = len(frame_results)
        consolidation_weights = config.get('consolidation_weights', {
            'mean_score': 0.30,
            'max_score': 0.20,
            'consistency': 0.30,
            'frequency': 0.20
        })
        
        for model_key, stats in model_stats.items():
            scores = stats['scores']
            
            # Métricas básicas
            stats['mean_score'] = sum(scores) / len(scores)
            stats['median_score'] = sorted(scores)[len(scores) // 2]
            stats['std_score'] = np.std(scores) if len(scores) > 1 else 0
            
            # Métricas de frecuencia
            stats['frequency_ratio'] = stats['detection_frequency'] / total_frames
            
            # Consistencia temporal
            frame_indices = sorted(stats['frame_indices'])
            if len(frame_indices) > 1:
                frame_gaps = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]
                stats['temporal_spread'] = max(frame_indices) - min(frame_indices)
                stats['avg_frame_gap'] = sum(frame_gaps) / len(frame_gaps)
                stats['consistency_score'] = min(1.0, 1.0 / (1.0 + stats['std_score']))
            else:
                stats['temporal_spread'] = 0
                stats['avg_frame_gap'] = 0
                stats['consistency_score'] = 1.0
            
            # Score final consolidado con pesos configurables
            stats['final_score'] = (
                stats['mean_score'] * consolidation_weights['mean_score'] +
                stats['max_score'] * consolidation_weights['max_score'] +
                stats['consistency_score'] * consolidation_weights['consistency'] +
                stats['frequency_ratio'] * consolidation_weights['frequency']
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
                'detection_method': 'automatic_ai_detection',
                'statistical_metrics': {
                    'median_score': round(stats['median_score'], 3),
                    'std_deviation': round(stats['std_score'], 3),
                    'score_range': round(stats['max_score'] - stats['min_score'], 3)
                }
            })
        
        # Confianza general
        overall_confidence = sorted_models[0][1]['final_score'] if sorted_models else 0.0
        
        logger.info(f"🔍 DEBUG - Overall confidence: {overall_confidence:.3f} ({overall_confidence*100:.1f}%)")
        logger.info(f"🔍 DEBUG - Umbral requerido: {settings.MIN_CONFIDENCE_FOR_TRAINING:.3f} ({settings.MIN_CONFIDENCE_FOR_TRAINING*100:.1f}%)")
        logger.info(f"🔍 DEBUG - ¿Califica para entrenamiento?: {overall_confidence >= settings.MIN_CONFIDENCE_FOR_TRAINING}")
        
        if sorted_models:
            best_model = sorted_models[0]
            logger.info(f"🔍 DEBUG - Mejor producto: {best_model[1]['product'].get('brand')} {best_model[1]['product'].get('model_name')}")
            logger.info(f"🔍 DEBUG - Frecuencia detección: {best_model[1]['frequency_ratio']:.2f}")
            logger.info(f"🔍 DEBUG - Score promedio: {best_model[1]['mean_score']:.3f}")
        
        # Información AI detallada
        ai_info = {
            "processing_method": "ultra_optimized_multi_frame",
            "strategy_used": config['strategy'],
            "consolidation_method": "weighted_statistical_analysis",
            "quality_filters": list(config.get('quality_config', {}).keys()),
            "batch_processing": True,
            "parallel_optimization": settings.ENABLE_PARALLEL_PROCESSING,
            "models_analyzed": len(model_stats),
            "statistical_confidence": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.5 else "low",
            "consolidation_weights": consolidation_weights,
            "frames_with_detections": len([f for f in frame_results if f.get('similar_products')])
        }
        
        return {
            "detected_products": detected_products,
            "overall_confidence": round(overall_confidence, 3),
            "frames_analyzed": len(frame_results),
            "models_considered": len(model_stats),
            "ai_info": ai_info,
            "should_add_to_training": overall_confidence >= settings.MIN_CONFIDENCE_FOR_TRAINING,
            "consolidation_type": "automatic_ai_detection"
        }
        
    def _get_360_product_config(self, duration: float) -> Dict[str, Any]:
        """🎯 Configuración específica para productos 360° completos"""
        
        # Calcular frames óptimos basado en duración
        optimal_frames = min(25, max(15, int(duration * 12)))  # ~12 frames por segundo
        
        return {
            'strategy': 'comprehensive_360_sampling',
            'interval_seconds': 0.1,  # Frame cada 100ms
            'max_frames': optimal_frames,
            'quality_config': {
                'min_brightness': 5,        # MUY permisivo
                'min_variance': 15,         # MUY permisivo  
                'quality_threshold': 0.15,  # MUY permisivo
                'focus_weight': 0.30,
                'brightness_weight': 0.20,
                'contrast_weight': 0.25,
                'motion_weight': 0.25
            },
            'consolidation_weights': {
                'mean_score': 0.30,     # Menos peso al promedio individual
                'max_score': 0.20,      # Menos peso al mejor frame
                'consistency': 0.30,    # MÁS peso a consistencia entre frames
                'frequency': 0.20       # MÁS peso a frecuencia de detección
            },
            'training_percentage': settings.TRAINING_FRAMES_PERCENTAGE,  # 80%
            'min_vectors_required': settings.MIN_VECTORS_FOR_PRODUCT,   # 10
            'priority': 'comprehensive_360_coverage'
        }
    
    async def _add_training_data_selective(
        self,
        frames: List[np.ndarray],
        consolidated_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """📚 Entrenamiento selectivo - SOPORTE 360° ESCALABLE"""
        
        detected_products = consolidated_result.get("detected_products", [])
        if not detected_products:
            logger.warning("⚠️ No hay productos detectados para entrenamiento")
            return False
        
        best_product = detected_products[0]
        
        # Verificar confianza
        confidence_threshold = settings.MIN_CONFIDENCE_FOR_TRAINING * 100
        if best_product['confidence'] < confidence_threshold:
            logger.info(f"⚠️ Confianza insuficiente: {best_product['confidence']:.1f}% < {confidence_threshold:.1f}%")
            return False
        
        # 🆕 SELECCIÓN 360° ESCALABLE
        config_tier = consolidated_result.get('ai_info', {}).get('config_tier', 'unknown')
        training_percentage = consolidated_result.get('ai_info', {}).get('training_percentage', 0.8)
        min_vectors_required = consolidated_result.get('ai_info', {}).get('min_vectors_required', 10)
        
        num_frames_for_training = max(min_vectors_required, int(len(frames) * training_percentage))
        
        # Distribuir frames uniformemente (no solo los primeros)
        if len(frames) > num_frames_for_training:
            step = len(frames) // num_frames_for_training
            selected_frames = frames[::step][:num_frames_for_training]
        else:
            selected_frames = frames
        
        logger.info(f"🎯 MODO 360° ESCALABLE [{config_tier}]: Entrenando con {len(selected_frames)} frames ({training_percentage*100:.0f}% del total)")
        
        training_success_count = 0
        
        for i, frame in enumerate(selected_frames):
            try:
                embedding = await self.embedding_service.get_image_embedding_from_array(frame)
                
                training_metadata = {
                    "brand": best_product.get("brand"),
                    "model_name": best_product.get("model_name"),
                    "color": best_product.get("color"),
                    "confidence": float(best_product.get("confidence", 0) / 100.0),
                    "source": f"360_scalable_{config_tier}_training",  # 🆕 Identificar tier
                    "warehouse_id": int(metadata.get("warehouse_id", 0)),
                    "admin_id": int(metadata.get("admin_id", 0)),
                    "frame_index": int(i),
                    "training_session": str(metadata.get("job_db_id", "unknown")),
                    "optimization_level": f"360_scalable_{config_tier}",  # 🆕 Nivel específico
                    "config_tier": config_tier,  # 🆕 Metadata del tier
                    "angle_coverage": f"frame_{i}_of_{len(selected_frames)}",
                    "training_timestamp": int(time.time())
                }
                
                vector_id = f"360s_{config_tier}_{metadata.get('job_db_id', 'unknown')}_{i}_{int(time.time() * 1000)}"
                
                success = await self.pinecone_service.upsert_vector(
                    vector_id, embedding, training_metadata
                )
                
                if success:
                    training_success_count += 1
                    logger.debug(f"📚 Frame 360° {i+1} [{config_tier}] agregado - Vector: {vector_id}")
                
                await asyncio.sleep(0.03)
                
            except Exception as e:
                logger.error(f"❌ Error agregando frame 360° {i}: {e}")
                continue
        
        success_rate = training_success_count / len(selected_frames)
        
        logger.info(f"✅ ENTRENAMIENTO 360° [{config_tier}] completado: {training_success_count}/{len(selected_frames)} frames ({success_rate:.1%})")
        
        if training_success_count >= min_vectors_required:
            logger.info(f"🎯 ¡PRODUCTO 360° LISTO! {training_success_count} vectores agregados (mínimo: {min_vectors_required})")
            return True
        else:
            logger.warning(f"⚠️ Entrenamiento 360° insuficiente: {training_success_count} < {min_vectors_required}")
            return False
    
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