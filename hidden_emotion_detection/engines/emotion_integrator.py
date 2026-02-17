#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情绪整合引擎
负责整合主引擎和辅助引擎的情绪分析结果，生成最终事件
"""

import logging
import time
from typing import Dict, List, Optional, Any
import threading

from hidden_emotion_detection.config import ConfigManager
from hidden_emotion_detection.core.event_bus import EventBus, Event, EventType
from hidden_emotion_detection.core.data_types import EmotionResult
from hidden_emotion_detection.utils.logger_util import get_module_logger

# 设置日志
logger = get_module_logger('emotion_integrator', 'logs/emotion_integrator.log')

class EmotionIntegrator:
    """情绪整合引擎，负责整合主引擎和辅助引擎的情绪分析结果，生成最终事件"""
    
    _instance = None  # 单例模式实例
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(EmotionIntegrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: ConfigManager, event_bus: EventBus):
        """初始化情绪整合引擎"""
        if getattr(self, '_initialized', False):
            return
            
        self.config = config
        self.event_bus = event_bus
        
        # 缓存主引擎和AU辅助引擎的最近结果
        self.macro_raw_results = {}  # {face_id: (timestamp, EmotionResult)}
        self.macro_au_results = {}   # {face_id: (timestamp, EmotionResult)}
        
        self.micro_raw_results = {}  # {face_id: (timestamp, EmotionResult)}
        self.micro_au_results = {}   # {face_id: (timestamp, EmotionResult)}
        
        # 配置参数
        self.integration_interval = self.config.get("integration.interval", 0.5)  # 整合间隔，秒
        self.raw_result_valid_time = self.config.get("integration.raw_result_valid_time", 2.0)  # 主引擎结果有效时间，秒
        self.au_result_valid_time = self.config.get("integration.au_result_valid_time", 2.0)  # AU辅助引擎结果有效时间，秒
        
        # 线程控制
        self.is_running = False
        self.integration_thread = None
        self.stop_event = threading.Event()
        
        # 订阅事件
        if self.event_bus:
            self.event_bus.subscribe(EventType.RAW_MACRO_EMOTION_ANALYZED, self._on_raw_macro_emotion_analyzed)
            self.event_bus.subscribe(EventType.AU_MACRO_EMOTION_ANALYZED, self._on_au_macro_emotion_analyzed)
            
            self.event_bus.subscribe(EventType.RAW_MICRO_EMOTION_ANALYZED, self._on_raw_micro_emotion_analyzed)
            self.event_bus.subscribe(EventType.AU_MICRO_EMOTION_ANALYZED, self._on_au_micro_emotion_analyzed)
            
            logger.info("EmotionIntegrator subscribed to emotion events.")
        
        self._initialized = True
        logger.info("EmotionIntegrator initialized.")
    
    def start(self):
        """启动情绪整合引擎"""
        if self.is_running:
            logger.warning("EmotionIntegrator already running.")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        logger.info("EmotionIntegrator started.")
        
    def stop(self):
        """停止情绪整合引擎"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=1.0)
            
        self._clear_all_caches()
        logger.info("EmotionIntegrator stopped.")
    
    def _clear_all_caches(self):
        """清空所有缓存"""
        self.macro_raw_results.clear()
        self.macro_au_results.clear()
        self.micro_raw_results.clear()
        self.micro_au_results.clear()
    
    def _on_raw_macro_emotion_analyzed(self, event: Event):
        """处理主宏观引擎事件"""
        if not self.is_running or event.type != EventType.RAW_MACRO_EMOTION_ANALYZED:
            return
            
        try:
            result = event.data.get("result")
            if result and isinstance(result, EmotionResult) and hasattr(result, "face_id"):
                face_id = result.face_id
                self.macro_raw_results[face_id] = (time.time(), result)
                logger.debug(f"Cached RAW_MACRO result for face_id {face_id}: {result.emotion_type} ({result.probability:.2f})")
        except Exception as e:
            logger.error(f"Error processing RAW_MACRO_EMOTION_ANALYZED event: {e}", exc_info=True)
    
    def _on_au_macro_emotion_analyzed(self, event: Event):
        """处理AU辅助宏观引擎事件"""
        if not self.is_running or event.type != EventType.AU_MACRO_EMOTION_ANALYZED:
            return
            
        try:
            result = event.data.get("result")
            if result and isinstance(result, EmotionResult) and hasattr(result, "face_id"):
                face_id = result.face_id
                self.macro_au_results[face_id] = (time.time(), result)
                logger.debug(f"Cached AU_MACRO result for face_id {face_id}: {result.emotion_type} ({result.probability:.2f})")
        except Exception as e:
            logger.error(f"Error processing AU_MACRO_EMOTION_ANALYZED event: {e}", exc_info=True)
    
    def _on_raw_micro_emotion_analyzed(self, event: Event):
        """处理主微表情引擎事件"""
        if not self.is_running or event.type != EventType.RAW_MICRO_EMOTION_ANALYZED:
            return
            
        try:
            result = event.data.get("result")
            if result and isinstance(result, EmotionResult) and hasattr(result, "face_id"):
                face_id = result.face_id
                self.micro_raw_results[face_id] = (time.time(), result)
                logger.debug(f"Cached RAW_MICRO result for face_id {face_id}: {result.emotion_type} ({result.probability:.2f})")
        except Exception as e:
            logger.error(f"Error processing RAW_MICRO_EMOTION_ANALYZED event: {e}", exc_info=True)
    
    def _on_au_micro_emotion_analyzed(self, event: Event):
        """处理AU辅助微表情引擎事件"""
        if not self.is_running or event.type != EventType.AU_MICRO_EMOTION_ANALYZED:
            return
            
        try:
            result = event.data.get("result")
            if result and isinstance(result, EmotionResult) and hasattr(result, "face_id"):
                face_id = result.face_id
                self.micro_au_results[face_id] = (time.time(), result)
                logger.debug(f"Cached AU_MICRO result for face_id {face_id}: {result.emotion_type} ({result.probability:.2f})")
        except Exception as e:
            logger.error(f"Error processing AU_MICRO_EMOTION_ANALYZED event: {e}", exc_info=True)
    
    def _integration_loop(self):
        """情绪整合主循环"""
        try:
            while self.is_running and not self.stop_event.is_set():
                # 整合宏观情绪
                self._integrate_macro_emotions()
                
                # 整合微表情
                self._integrate_micro_emotions()
                
                # 清理过期数据
                self._cleanup_expired_results()
                
                # 间隔时间
                time.sleep(self.integration_interval)
        except Exception as e:
            logger.error(f"Error in integration loop: {e}", exc_info=True)
    
    def _integrate_macro_emotions(self):
        """整合宏观情绪"""
        current_time = time.time()
        face_ids = set(self.macro_raw_results.keys()) | set(self.macro_au_results.keys())
        
        for face_id in face_ids:
            raw_data = self.macro_raw_results.get(face_id)
            au_data = self.macro_au_results.get(face_id)
            
            raw_valid = raw_data and (current_time - raw_data[0]) <= self.raw_result_valid_time
            au_valid = au_data and (current_time - au_data[0]) <= self.au_result_valid_time
            
            # 整合策略
            final_result = None
            if raw_valid and au_valid:
                # 两个结果都有效，根据置信度选择
                raw_timestamp, raw_result = raw_data
                au_timestamp, au_result = au_data
                
                if au_result.probability > raw_result.probability * 1.2:  # AU结果置信度显著更高
                    final_result = au_result
                    logger.info(f"Chosen AU macro result for face {face_id}: {au_result.emotion_type} ({au_result.probability:.2f})")
                else:
                    final_result = raw_result
                    logger.info(f"Chosen RAW macro result for face {face_id}: {raw_result.emotion_type} ({raw_result.probability:.2f})")
            elif raw_valid:
                # 只有主引擎结果有效
                final_result = raw_data[1]
                logger.debug(f"Only RAW macro result available for face {face_id}")
            elif au_valid:
                # 只有AU辅助结果有效
                final_result = au_data[1]
                logger.debug(f"Only AU macro result available for face {face_id}")
            
            # 发布最终结果
            if final_result:
                self.event_bus.publish(
                    EventType.MACRO_EMOTION_ANALYZED, 
                    {
                        "result": final_result, 
                        "face_id": face_id,
                        "source": "EmotionIntegrator"
                    },
                    source=self.__class__.__name__
                )
                logger.info(f"Published MACRO_EMOTION_ANALYZED for face {face_id}: {final_result.emotion_type} ({final_result.probability:.2f})")
    
    def _integrate_micro_emotions(self):
        """整合微表情"""
        current_time = time.time()
        face_ids = set(self.micro_raw_results.keys()) | set(self.micro_au_results.keys())
        
        for face_id in face_ids:
            raw_data = self.micro_raw_results.get(face_id)
            au_data = self.micro_au_results.get(face_id)
            
            raw_valid = raw_data and (current_time - raw_data[0]) <= self.raw_result_valid_time
            au_valid = au_data and (current_time - au_data[0]) <= self.au_result_valid_time
            
            # 整合策略
            final_result = None
            if raw_valid and au_valid:
                # 两个结果都有效，根据置信度选择
                raw_timestamp, raw_result = raw_data
                au_timestamp, au_result = au_data
                
                if au_result.probability > raw_result.probability * 1.2:  # AU结果置信度显著更高
                    final_result = au_result
                    logger.info(f"Chosen AU micro result for face {face_id}: {au_result.emotion_type} ({au_result.probability:.2f})")
                else:
                    final_result = raw_result
                    logger.info(f"Chosen RAW micro result for face {face_id}: {raw_result.emotion_type} ({raw_result.probability:.2f})")
            elif raw_valid:
                # 只有主引擎结果有效
                final_result = raw_data[1]
                logger.debug(f"Only RAW micro result available for face {face_id}")
            elif au_valid:
                # 只有AU辅助结果有效
                final_result = au_data[1]
                logger.debug(f"Only AU micro result available for face {face_id}")
            
            # 发布最终结果
            if final_result:
                # 确保是微表情结果
                if not hasattr(final_result, 'is_micro_expression'):
                    setattr(final_result, 'is_micro_expression', True)
                    
                self.event_bus.publish(
                    EventType.MICRO_EMOTION_ANALYZED, 
                    {
                        "result": final_result, 
                        "face_id": face_id,
                        "source": "EmotionIntegrator",
                        "raw_probabilities": getattr(final_result, "raw_probabilities", None)
                    },
                    source=self.__class__.__name__
                )
                logger.info(f"Published MICRO_EMOTION_ANALYZED for face {face_id}: {final_result.emotion_type} ({final_result.probability:.2f})")
    
    def _cleanup_expired_results(self):
        """清理过期的结果"""
        current_time = time.time()
        
        # 清理宏观情绪结果
        for face_id in list(self.macro_raw_results.keys()):
            if current_time - self.macro_raw_results[face_id][0] > self.raw_result_valid_time * 2:
                del self.macro_raw_results[face_id]
                
        for face_id in list(self.macro_au_results.keys()):
            if current_time - self.macro_au_results[face_id][0] > self.au_result_valid_time * 2:
                del self.macro_au_results[face_id]
        
        # 清理微表情结果
        for face_id in list(self.micro_raw_results.keys()):
            if current_time - self.micro_raw_results[face_id][0] > self.raw_result_valid_time * 2:
                del self.micro_raw_results[face_id]
                
        for face_id in list(self.micro_au_results.keys()):
            if current_time - self.micro_au_results[face_id][0] > self.au_result_valid_time * 2:
                del self.micro_au_results[face_id] 