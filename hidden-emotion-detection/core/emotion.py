#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情绪数据模型和状态表示模块
定义系统中所有情绪相关的数据结构和状态转换逻辑
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

# 使用从data_types导入的EmotionType，保持一致性
from .data_types import EmotionType

@dataclass
class EmotionState:
    """情绪状态类，表示某一时刻的情绪状态"""
    
    emotion_type: EmotionType  # 情绪类型
    intensity: float  # 情绪强度（0.0-1.0）
    confidence: float  # 置信度（0.0-1.0）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __str__(self) -> str:
        """格式化为字符串表示"""
        return f"{self.emotion_type}(强度={self.intensity:.2f}, 置信度={self.confidence:.2f})"
    
    def is_stronger_than(self, other: 'EmotionState') -> bool:
        """判断情绪强度是否比另一情绪状态更强"""
        return self.intensity > other.intensity
    
    def is_same_type(self, other: 'EmotionState') -> bool:
        """判断是否是相同类型的情绪"""
        return self.emotion_type == other.emotion_type
    
    def blend_with(self, other: 'EmotionState', weight: float = 0.5) -> 'EmotionState':
        """与另一情绪状态混合（用于平滑过渡）"""
        if self.emotion_type != other.emotion_type:
            # 不同类型情绪不能混合
            return self if weight < 0.5 else other
        
        # 混合强度和置信度
        new_intensity = self.intensity * (1 - weight) + other.intensity * weight
        new_confidence = self.confidence * (1 - weight) + other.confidence * weight
        
        return EmotionState(
            emotion_type=self.emotion_type,
            intensity=new_intensity,
            confidence=new_confidence,
            timestamp=max(self.timestamp, other.timestamp)
        )


@dataclass
class EmotionTracker:
    """情绪追踪器，跟踪情绪状态随时间的变化"""
    
    history: List[EmotionState] = field(default_factory=list)  # 情绪历史记录
    max_history: int = 100  # 历史记录最大长度
    
    def add(self, state: EmotionState) -> None:
        """添加新的情绪状态"""
        self.history.append(state)
        
        # 如果超过最大长度，删除最早的记录
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_current(self) -> Optional[EmotionState]:
        """获取当前情绪状态"""
        if not self.history:
            return None
        return self.history[-1]
    
    def get_dominant(self, window_size: int = None) -> Optional[EmotionState]:
        """获取一段时间内的主要情绪状态"""
        if not self.history:
            return None
        
        # 确定窗口大小
        if window_size is None or window_size > len(self.history):
            window_size = len(self.history)
        
        # 获取最近的窗口
        recent = self.history[-window_size:]
        
        # 按情绪类型分组并计算平均强度
        emotion_groups = {}
        for state in recent:
            if state.emotion_type not in emotion_groups:
                emotion_groups[state.emotion_type] = []
            emotion_groups[state.emotion_type].append(state)
        
        # 计算每种情绪的平均强度和置信度
        emotion_scores = {}
        for emotion_type, states in emotion_groups.items():
            avg_intensity = sum(s.intensity for s in states) / len(states)
            avg_confidence = sum(s.confidence for s in states) / len(states)
            count = len(states)
            # 计算得分 = 平均强度 * 平均置信度 * sqrt(出现次数/总样本数)
            score = avg_intensity * avg_confidence * (count / window_size) ** 0.5
            emotion_scores[emotion_type] = (score, avg_intensity, avg_confidence)
        
        # 找出得分最高的情绪
        dominant_type = max(emotion_scores.keys(), key=lambda k: emotion_scores[k][0])
        score, intensity, confidence = emotion_scores[dominant_type]
        
        return EmotionState(
            emotion_type=dominant_type,
            intensity=intensity,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def has_change(self, threshold: float = 0.3) -> bool:
        """检测是否有显著的情绪变化"""
        if len(self.history) < 2:
            return False
        
        current = self.history[-1]
        previous = self.history[-2]
        
        # 情绪类型变化
        if current.emotion_type != previous.emotion_type:
            return True
        
        # 情绪强度变化超过阈值
        if abs(current.intensity - previous.intensity) > threshold:
            return True
        
        return False
    
    def clear(self) -> None:
        """清空历史记录"""
        self.history.clear()


@dataclass
class HiddenEmotionState:
    """隐藏情绪状态类，表示表面情绪和隐藏情绪的关系"""
    
    surface: EmotionState  # 表面情绪
    hidden: EmotionState  # 隐藏情绪
    conflict_score: float  # 冲突程度（0.0-1.0）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    evidence: List[str] = field(default_factory=list)  # 证据描述
    
    def __str__(self) -> str:
        """格式化为字符串表示"""
        return f"表面:{self.surface}, 隐藏:{self.hidden}, 冲突:{self.conflict_score:.2f}"
    
    def has_conflict(self, threshold: float = 0.5) -> bool:
        """判断是否存在明显的情绪冲突"""
        return (self.conflict_score >= threshold and 
                self.surface.emotion_type != self.hidden.emotion_type)


@dataclass
class EmotionProfile:
    """情绪特征档案，描述个体的情绪表达特点"""
    
    # 各情绪类型的基线强度
    baseline: Dict[EmotionType, float] = field(default_factory=dict)
    
    # 情绪表达的个人偏好（哪些情绪更容易/难以表达）
    expression_bias: Dict[EmotionType, float] = field(default_factory=dict)
    
    # 情绪掩饰模式（哪些情绪倾向于被掩饰为哪些情绪）
    masking_patterns: Dict[EmotionType, Dict[EmotionType, float]] = field(default_factory=dict)
    
    # 情绪切换速度（从一种情绪到另一种情绪的转换速度）
    transition_rates: Dict[Tuple[EmotionType, EmotionType], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后设置默认值"""
        # 为所有情绪类型设置默认基线
        for emotion in EmotionType:
            if emotion not in self.baseline:
                self.baseline[emotion] = 0.1  # 默认基线强度
            
            if emotion not in self.expression_bias:
                self.expression_bias[emotion] = 1.0  # 默认无偏好
            
            if emotion not in self.masking_patterns:
                self.masking_patterns[emotion] = {}
                # 默认掩饰模式：情绪倾向于被掩饰为中性
                if emotion != EmotionType.NEUTRAL:
                    self.masking_patterns[emotion][EmotionType.NEUTRAL] = 0.7
    
    def update_from_observation(self, state: EmotionState) -> None:
        """根据观察到的情绪状态更新情绪特征"""
        emotion_type = state.emotion_type
        intensity = state.intensity
        
        # 更新基线（使用滑动平均）
        current_baseline = self.baseline.get(emotion_type, 0.1)
        self.baseline[emotion_type] = current_baseline * 0.9 + intensity * 0.1
    
    def predict_masking(self, true_emotion: EmotionType) -> EmotionType:
        """预测给定真实情绪可能会被掩饰为的表面情绪"""
        if true_emotion not in self.masking_patterns:
            return EmotionType.NEUTRAL
        
        patterns = self.masking_patterns[true_emotion]
        if not patterns:
            return EmotionType.NEUTRAL
        
        # 返回最可能的掩饰情绪
        return max(patterns.keys(), key=lambda k: patterns[k])
    
    def get_expression_intensity(self, emotion_type: EmotionType, internal_intensity: float) -> float:
        """根据内部情绪强度和表达偏好计算表达强度"""
        bias = self.expression_bias.get(emotion_type, 1.0)
        baseline = self.baseline.get(emotion_type, 0.1)
        
        # 计算表达强度，考虑个人偏好和基线
        return (internal_intensity - baseline) * bias + baseline 