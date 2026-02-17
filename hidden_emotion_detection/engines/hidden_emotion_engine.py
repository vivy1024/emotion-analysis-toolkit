#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
隐藏情绪分析引擎
基于微表情和宏观表情差异检测隐藏情绪，结合AU模型提供科学依据
"""

import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import json

from hidden_emotion_detection.config import config_manager # 导入全局配置实例
from hidden_emotion_detection.core.data_types import (
    EmotionResult, EmotionType, Event, EventType, AUResult, HiddenEmotionResult
)
from hidden_emotion_detection.core.event_bus import EventBus

# 配置日志
logger = logging.getLogger("enhance_hidden.engines.HiddenEmotionEngine")

class HiddenEmotionEngine:
    """
    隐藏情绪分析引擎
    基于微表情和宏观表情的差异，结合AU动作单元分析，检测是否存在隐藏情绪
    """
    
    _instance = None  # 单例实例
    
    # 情绪矛盾关系定义 - 基于学术研究
    # 以下定义了哪些情绪组合可能表示隐藏
    # 例如：如果宏观表情是中性，但微表情显示愤怒，可能是隐藏的愤怒
    EMOTION_CONTRADICTIONS = {
        # 宏观表情 -> 可能隐藏的微表情
        EmotionType.NEUTRAL: {
            EmotionType.ANGER, EmotionType.DISGUST, EmotionType.FEAR,
            EmotionType.SADNESS, EmotionType.SURPRISE
        },
        EmotionType.HAPPINESS: {
            EmotionType.ANGER, EmotionType.DISGUST, EmotionType.FEAR, 
            EmotionType.SADNESS, EmotionType.CONFUSION
        },
        EmotionType.SADNESS: {
            EmotionType.ANGER, EmotionType.DISGUST
        },
        EmotionType.SURPRISE: {
            EmotionType.FEAR, EmotionType.DISGUST, EmotionType.CONFUSION
        },
        EmotionType.ANGER: {
            EmotionType.FEAR, EmotionType.SADNESS
        },
        EmotionType.FEAR: {
            EmotionType.ANGER, EmotionType.DISGUST
        },
        EmotionType.DISGUST: {
            EmotionType.FEAR, EmotionType.SADNESS
        }
    }
    
    # AU与隐藏情绪关联 - 基于学术研究
    # 定义特定AU可能表明存在的隐藏情绪
    HIDDEN_EMOTION_AU_MAPPING = {
        # 愤怒的微表情特征
        EmotionType.ANGER: {
            "aus": [4, 7, 23, 24],
            "descriptions": {
                4: "眉头皱起 - 可能表明抑制愤怒",
                7: "眼睑紧合 - 常见于隐藏愤怒时的面部紧张",
                23: "嘴唇紧闭 - 可能是压抑言语表达愤怒的表现",
                24: "嘴唇压紧 - 表明情绪压抑和克制"
            }
        },
        # 厌恶的微表情特征
        EmotionType.DISGUST: {
            "aus": [9, 10, 17],
            "descriptions": {
                9: "鼻皱 - 微表情中厌恶的典型特征",
                10: "上唇上提 - 厌恶情绪的微妙表现",
                17: "下巴下颌抬升 - 表明控制厌恶反应"
            }
        },
        # 恐惧的微表情特征
        EmotionType.FEAR: {
            "aus": [1, 2, 5, 20, 26],
            "descriptions": {
                1: "内眉上扬 - 恐惧情绪的微表情特征",
                2: "外眉上扬 - 与惊讶混合的恐惧表现",
                5: "上眼睑提升 - 警觉性增强的表现",
                20: "嘴角横向拉伸 - 恐惧时的紧张表现",
                26: "下颌下垂 - 短暂的恐惧反应"
            }
        },
        # 悲伤的微表情特征
        EmotionType.SADNESS: {
            "aus": [1, 4, 15, 17],
            "descriptions": {
                1: "内眉上扬 - 悲伤的主要特征",
                4: "眉头皱起 - 悲伤和痛苦的混合表现",
                15: "嘴角下垂 - 短暂出现的悲伤表现",
                17: "下巴下颌抬升 - 试图控制情绪波动"
            }
        },
        # 压抑的微表情特征 (使用CONFUSION代替REPRESSION)
        EmotionType.CONFUSION: {
            "aus": [4, 7, 15, 17, 23, 24],
            "descriptions": {
                4: "眉头皱起 - 情绪压抑的表现",
                7: "眼睑紧合 - 高度自我控制",
                15: "嘴角下垂 - 消极情绪的微表情",
                17: "下巴下颌抬升 - 试图维持面部控制",
                23: "嘴唇紧闭 - 控制情绪表达",
                24: "嘴唇压紧 - 强烈压抑情绪的表现"
            }
        }
    }
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(HiddenEmotionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化隐藏情绪分析引擎"""
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
            
        logger.info("初始化隐藏情绪分析引擎...")
        
        # 获取配置管理器实例
        self.config_manager = config_manager
        
        # 获取配置
        self.conflict_threshold = self.config_manager.get("hidden.conflict_threshold", 0.6)
        self.detection_threshold = self.config_manager.get("hidden.detection_threshold", 0.7)
        self.emotion_diff_weight = self.config_manager.get("hidden.emotion_diff_weight", 0.6)
        self.au_evidence_weight = self.config_manager.get("hidden.au_evidence_weight", 0.4)
        self.min_confidence = self.config_manager.get("hidden.min_confidence", 0.6)
        self.update_interval = self.config_manager.get("hidden.update_interval", 5)
        self.enabled = self.config_manager.get("hidden.enabled", True) # 读取 enabled 状态
        
        # 状态和结果存储
        self.is_running = False
        self.is_paused = False
        self.result_lock = threading.Lock()
        self.current_result = None
        self.processing_time = 0.0
        
        # 缓存当前的宏观和微观表情结果
        self.current_macro_result = None
        self.current_micro_result = None
        self.current_au_result = None
        self.current_face = None
        self.current_frame = None
        self.current_frame_id = 0
        
        # 处理时间跟踪
        self.processing_times = []
        
        # 获取事件总线实例
        self.event_bus = EventBus()
        
        # 注册配置变更回调
        self.config_manager.register_change_callback("hidden", self._on_config_changed)
        
        # 注册事件处理
        # 修改：同时订阅宏观和微观情绪事件
        self.event_bus.subscribe(EventType.MACRO_EMOTION_ANALYZED, self._on_macro_emotion_analyzed)
        self.event_bus.subscribe(EventType.MICRO_EMOTION_ANALYZED, self._on_micro_emotion_analyzed)
        self.event_bus.subscribe(EventType.AU_ANALYZED, self._on_au_analyzed)
        
        self._initialized = True
        logger.info("隐藏情绪分析引擎初始化完成")
    
    def _on_config_changed(self, path: str, value: Any):
        """配置变更回调"""
        if path == "hidden.detection_threshold":
            logger.info(f"隐藏情绪检测阈值已更新为: {value}")
            self.detection_threshold = value
        elif path == "hidden.emotion_diff_weight":
            logger.info(f"情绪差异权重已更新为: {value}")
            self.emotion_diff_weight = value
        elif path == "hidden.au_evidence_weight":
            logger.info(f"AU证据权重已更新为: {value}")
            self.au_evidence_weight = value
        elif path == "hidden.min_confidence":
            logger.info(f"最小置信度已更新为: {value}")
            self.min_confidence = value
        elif path == "hidden.update_interval":
            logger.info(f"更新间隔已更新为: {value}")
            self.update_interval = value
        elif path == "hidden.enabled": # 处理 enabled 配置变更
             logger.info(f"隐藏情绪引擎已{'启用' if value else '禁用'}")
             self.enabled = value
             if not value: # 禁用时重置状态
                  self.current_macro_result = None
                  self.current_micro_result = None
                  self.current_au_result = None
                  with self.result_lock:
                       self.current_result = None
    
    def _on_macro_emotion_analyzed(self, event: Event):
        """宏观情绪分析事件处理"""
        if not self.enabled or not self.is_running or self.is_paused:
            return
            
        # 提取事件数据
        event_data = event.data
        if not event_data or "result" not in event_data or "face" not in event_data:
            return
            
        # 更新当前宏观情绪结果、人脸和帧
        self.current_macro_result = event_data["result"]
        self.current_face = event_data["face"]
        self.current_frame = event_data.get("frame")
        self.current_frame_id = event_data.get("frame_id", 0)
        
        logger.debug(f"收到宏观情绪分析结果: {self.current_macro_result}")
        
        # 检查是否可以进行隐藏情绪分析
        self._check_and_analyze()
    
    def _on_micro_emotion_analyzed(self, event: Event):
        """微观情绪分析事件处理"""
        if not self.enabled or not self.is_running or self.is_paused:
            return
            
        # 提取事件数据
        event_data = event.data
        if not event_data or "result" not in event_data or "face" not in event_data:
            return
            
        # 更新当前微观情绪结果、人脸和帧
        self.current_micro_result = event_data["result"]
        self.current_face = event_data["face"]
        self.current_frame = event_data.get("frame")
        self.current_frame_id = event_data.get("frame_id", 0)
        
        logger.debug(f"收到微观情绪分析结果: {self.current_micro_result}")
        
        # 检查是否可以进行隐藏情绪分析
        self._check_and_analyze()
    
    def _on_au_analyzed(self, event: Event):
        """AU分析事件处理"""
        if not self.enabled or not self.is_running or self.is_paused:
            return
            
        # 提取事件数据
        event_data = event.data
        if not event_data or "result" not in event_data or "face" not in event_data:
            return
            
        # 更新当前AU结果、人脸和帧
        self.current_au_result = event_data["result"]
        self.current_face = event_data["face"]
        self.current_frame = event_data.get("frame")
        self.current_frame_id = event_data.get("frame_id", 0)
        
        logger.debug(f"收到AU分析结果: {self.current_au_result}")
        
        # 检查是否可以进行隐藏情绪分析
        self._check_and_analyze()
    
    def _check_and_analyze(self):
        """检查是否满足隐藏情绪分析条件，并执行分析"""
        # 确保所有必要的数据都准备好了
        if (self.current_macro_result and self.current_micro_result and 
            self.current_au_result and self.current_face and
            self.current_frame_id % self.update_interval == 0):
            
            logger.debug("所有数据已准备就绪，开始分析隐藏情绪")
            
            # 在线程中进行隐藏情绪分析
            threading.Thread(
                target=self._analyze_hidden_emotion_threaded,
                daemon=True
            ).start()
    
    def _analyze_hidden_emotion_threaded(self):
        """在线程中执行隐藏情绪分析"""
        try:
            # 进行隐藏情绪分析
            result_dict = self._analyze_hidden_emotion()
            
            if result_dict:
                # 创建HiddenEmotionResult对象
                hidden_result = HiddenEmotionResult(
                    surface_emotion=EmotionType(result_dict["macro_emotion"]),
                    hidden_emotion=EmotionType(result_dict["hidden_emotion"]) if result_dict["hidden_emotion"] != "未检测到" else None,
                    surface_prob=self.current_macro_result.probability,
                    hidden_prob=result_dict["confidence"],
                    conflict_score=result_dict["confidence"],
                    timestamp=time.time()
                )
                
                # 保存到人脸对象
                if self.current_face:
                    self.current_face.hidden_emotion = hidden_result
                
                # 保存分析结果
                with self.result_lock:
                    self.current_result = result_dict
                
                # 发布隐藏情绪检测事件
                self.event_bus.publish(
                    EventType.HIDDEN_EMOTION_ANALYZED,
                    {
                        "face": self.current_face,
                        "result": hidden_result,
                        "frame_id": self.current_frame_id
                    }
                )
                
                logger.debug(f"隐藏情绪分析完成并发布事件: {hidden_result}")
        except Exception as e:
            logger.error(f"隐藏情绪分析线程错误: {e}")
            logger.debug("异常详情", exc_info=True)
    
    def _analyze_hidden_emotion(self) -> Optional[Dict]:
        """
        分析宏观表情和微表情差异，检测隐藏情绪
        
        Returns:
            Dict: 隐藏情绪分析结果
        """
        if not self.current_macro_result or not self.current_micro_result:
            return None
            
        try:
            # 计时开始
            start_time = time.time()
            
            # 获取宏观和微观情绪类型
            macro_emotion = self.current_macro_result.emotion_type
            micro_emotion = self.current_micro_result.emotion_type
            
            # 获取概率值
            macro_prob = self.current_macro_result.probability
            micro_prob = self.current_micro_result.probability
            
            # 初始化结果
            is_hidden_emotion = False
            hidden_emotion_type = None
            confidence = 0.0
            supporting_aus = []
            
            # 情绪差异分析
            emotion_diff_score = self._calculate_emotion_diff_score(macro_emotion, micro_emotion)
            
            # 确定是否存在情绪矛盾关系
            if macro_emotion in self.EMOTION_CONTRADICTIONS and micro_emotion in self.EMOTION_CONTRADICTIONS[macro_emotion]:
                # 微表情可能是隐藏情绪
                hidden_emotion_type = micro_emotion
                
                # 计算AU证据分数
                au_evidence_score = 0.0
                if self.current_au_result:
                    au_evidence_score, supporting_aus = self._calculate_au_evidence_score(
                        hidden_emotion_type, self.current_au_result)
                
                # 计算综合置信度
                confidence = (
                    self.emotion_diff_weight * emotion_diff_score + 
                    self.au_evidence_weight * au_evidence_score
                )
                
                # 判断是否达到检测阈值
                is_hidden_emotion = confidence > self.detection_threshold
            
            # 创建结果
            result = {
                "is_hidden": is_hidden_emotion,
                "confidence": float(confidence),
                "macro_emotion": macro_emotion.value,
                "micro_emotion": micro_emotion.value,
                "hidden_emotion": hidden_emotion_type.value if hidden_emotion_type else "未检测到",
                "supporting_aus": supporting_aus
            }
            
            # 更新处理时间
            self.processing_time = time.time() - start_time
            self.processing_times.append(self.processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"分析隐藏情绪失败: {e}")
            logger.debug("异常详情", exc_info=True)
            return None
    
    def _calculate_emotion_diff_score(self, macro_emotion: EmotionType, micro_emotion: EmotionType) -> float:
        """
        计算宏观情绪和微观情绪之间的差异分数
        
        Args:
            macro_emotion: 宏观情绪类型
            micro_emotion: 微观情绪类型
            
        Returns:
            float: 差异分数 [0, 1]
        """
        # 如果情绪类型相同，没有差异
        if macro_emotion == micro_emotion:
            return 0.0
            
        # 检查是否存在矛盾关系
        if macro_emotion in self.EMOTION_CONTRADICTIONS and micro_emotion in self.EMOTION_CONTRADICTIONS[macro_emotion]:
            # 根据不同情绪组合计算差异分数
            
            # 对于中性宏观表情和任何微观情绪，差异分数较高
            if macro_emotion == EmotionType.NEUTRAL:
                return 0.8
                
            # 对于高兴的宏观表情和负面微观情绪，差异更明显
            if macro_emotion == EmotionType.HAPPINESS and micro_emotion in [
                EmotionType.ANGER, EmotionType.DISGUST, EmotionType.FEAR, 
                EmotionType.SADNESS, EmotionType.CONFUSION
            ]:
                return 0.9
                
            # 一般情况的矛盾
            return 0.7
        
        # 无明确矛盾关系，但情绪不同，返回中等差异分数
        return 0.5
    
    def _calculate_au_evidence_score(self, hidden_emotion: EmotionType, au_result: AUResult) -> Tuple[float, List[Dict]]:
        """
        计算AU证据分数，评估AU是否支持存在隐藏情绪
        
        Args:
            hidden_emotion: 潜在的隐藏情绪类型
            au_result: AU分析结果
            
        Returns:
            Tuple[float, List[Dict]]: 分数和支持的AU列表
        """
        score = 0.0
        supporting_aus = []
        
        # 如果没有该情绪类型的AU映射，返回零分
        if hidden_emotion not in self.HIDDEN_EMOTION_AU_MAPPING:
            return score, supporting_aus
            
        # 获取与该隐藏情绪相关的AU列表
        emotion_aus = self.HIDDEN_EMOTION_AU_MAPPING[hidden_emotion]["aus"]
        au_descriptions = self.HIDDEN_EMOTION_AU_MAPPING[hidden_emotion]["descriptions"]
        
        # 统计激活的相关AU数量
        activated_aus_count = 0
        total_intensity = 0.0
        
        for au_id in emotion_aus:
            au_id_str = str(au_id)
            
            # 检查AU是否存在且激活
            if au_id_str in au_result.au_present and au_result.au_present.get(au_id_str, False):
                activated_aus_count += 1
                
                # 获取AU强度
                intensity = au_result.au_intensities.get(au_id_str, 0.0)
                total_intensity += intensity
                
                # 添加到支持列表
                supporting_aus.append({
                    "id": au_id_str,
                    "name": f"AU{au_id_str}",
                    "intensity": float(intensity),
                    "description": au_descriptions.get(au_id, "未知AU功能")
                })
        
        # 如果没有激活的相关AU，返回零分
        if not activated_aus_count:
            return 0.0, []
            
        # 计算分数：激活率 * 平均强度
        activation_ratio = activated_aus_count / len(emotion_aus)
        avg_intensity = total_intensity / activated_aus_count if activated_aus_count > 0 else 0
        
        score = activation_ratio * avg_intensity
        
        # 根据支持的AU强度排序
        supporting_aus.sort(key=lambda x: x["intensity"], reverse=True)
        
        return score, supporting_aus
    
    def start(self):
        """启动隐藏情绪分析引擎"""
        if self.is_running:
            logger.warning("隐藏情绪分析引擎已经在运行中")
            return
        
        self.is_running = True
        self.is_paused = False
        logger.info("隐藏情绪分析引擎已启动")
        
        # 通知引擎启动事件
        self.event_bus.publish(EventType.ENGINE_STARTED, {"engine": "hidden"})
    
    def stop(self):
        """停止隐藏情绪分析引擎"""
        if not self.is_running:
            logger.warning("隐藏情绪分析引擎未运行")
            return
        
        self.is_running = False
        logger.info("隐藏情绪分析引擎已停止")
        
        # 通知引擎停止事件
        self.event_bus.publish(EventType.ENGINE_STOPPED, {"engine": "hidden"})
    
    def pause(self):
        """暂停隐藏情绪分析引擎"""
        if not self.is_running:
            logger.warning("隐藏情绪分析引擎未运行")
            return
        
        self.is_paused = True
        logger.info("隐藏情绪分析引擎已暂停")
        
        # 通知引擎暂停事件
        self.event_bus.publish(EventType.ENGINE_PAUSED, {"engine": "hidden"})
    
    def resume(self):
        """恢复隐藏情绪分析引擎"""
        if not self.is_running:
            logger.warning("隐藏情绪分析引擎未运行")
            return
        
        self.is_paused = False
        logger.info("隐藏情绪分析引擎已恢复")
        
        # 通知引擎恢复事件
        self.event_bus.publish(EventType.ENGINE_RESUMED, {"engine": "hidden"})
    
    def get_current_result(self) -> Optional[Dict]:
        """获取当前隐藏情绪分析结果"""
        with self.result_lock:
            return self.current_result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计数据"""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        return {
            "processing_time": self.processing_time,
            "average_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def analyze(self, face) -> Optional[HiddenEmotionResult]:
        """
        分析宏观表情和微观表情的差异，检测隐藏情绪（分析特定人脸）
        
        Args:
            face: 人脸对象，包含宏观情绪和微观情绪结果
            
        Returns:
            HiddenEmotionResult: 隐藏情绪分析结果
        """
        if not self.is_running or self.is_paused:
            return None
        
        # 确保人脸包含必要的属性
        if not hasattr(face, 'macro_emotion') or not hasattr(face, 'micro_emotion'):
            logger.warning("人脸对象缺少宏观或微观情绪属性")
            return None
            
        # 确保有AU结果
        if not hasattr(face, 'au_result'):
            logger.warning("人脸对象缺少AU分析结果")
            return None
            
        # 更新当前结果
        self.current_macro_result = face.macro_emotion
        self.current_micro_result = face.micro_emotion
        self.current_au_result = face.au_result
        self.current_face = face
            
        # 分析隐藏情绪
        result_dict = self._analyze_hidden_emotion()
        
        if not result_dict:
            return None
            
        # 创建HiddenEmotionResult对象
        hidden_result = HiddenEmotionResult(
            surface_emotion=EmotionType(result_dict["macro_emotion"]),
            hidden_emotion=EmotionType(result_dict["hidden_emotion"]) if result_dict["hidden_emotion"] != "未检测到" else None,
            surface_prob=self.current_macro_result.probability,
            hidden_prob=result_dict["confidence"],
            conflict_score=result_dict["confidence"],
            timestamp=time.time()
        )
        
        # 发布隐藏情绪检测事件
        self.event_bus.publish(
            EventType.HIDDEN_EMOTION_ANALYZED,
            {
                "face": face,
                "result": hidden_result,
                "frame_id": self.current_frame_id
            }
        )
        
        return hidden_result