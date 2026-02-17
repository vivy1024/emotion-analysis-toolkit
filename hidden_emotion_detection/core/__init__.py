#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏表情检测系统 - 核心模块
包含系统核心数据结构、基础组件和管道定义
"""

from .data_types import (
    EmotionType, FaceDetection, FaceBox, FacePose, Landmarks,
    AUStateMap, AUResult, EmotionResult, HiddenEmotionResult,
    FrameResult, EventType, Event
)

from .emotion import (
    EmotionState, EmotionTracker, HiddenEmotionState, EmotionProfile
)

from .frame import Frame

from .event_bus import EventBus

from .pipeline import (
    PipelineStage, Pipeline, AsyncPipeline, FunctionStage,
    FilterStage, BranchPipeline, pipeline, function_stage
)

from .base import (
    Point, IEngine
)

# 创建全局事件总线实例
event_bus = EventBus()

__all__ = [
    # 数据类型
    'EmotionType', 'FaceDetection', 'FaceBox', 'FacePose', 'Landmarks',
    'AUStateMap', 'AUResult', 'EmotionResult', 'HiddenEmotionResult',
    'FrameResult', 'EventType', 'Event',
    
    # 情绪模型
    'EmotionState', 'EmotionTracker', 'HiddenEmotionState', 'EmotionProfile',
    
    # 帧模型
    'Frame',
    
    # 事件总线
    'EventBus', 'event_bus',
    
    # 管道
    'PipelineStage', 'Pipeline', 'AsyncPipeline', 'FunctionStage',
    'FilterStage', 'BranchPipeline', 'pipeline', 'function_stage',
    
    # 基础接口
    'Point', 'IEngine'
] 