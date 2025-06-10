#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频帧封装模块
定义视频帧数据结构及相关元数据处理功能
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .data_types import FaceDetection, EmotionResult, AUResult, HiddenEmotionResult


@dataclass
class Frame:
    """视频帧封装类，包含帧图像和元数据"""
    
    frame_id: int  # 帧ID
    image: Optional[np.ndarray] = None  # 帧图像数据
    timestamp: float = field(default_factory=time.time)  # 时间戳
    width: int = 0  # 图像宽度
    height: int = 0  # 图像高度
    channels: int = 0  # 图像通道数, 0表示未初始化或未知
    
    # 分析结果
    faces: Dict[int, FaceDetection] = field(default_factory=dict)  # 人脸检测结果 {face_id: FaceDetection}
    emotions: Dict[int, EmotionResult] = field(default_factory=dict)  # 情绪分析结果 {face_id: EmotionResult}
    aus: Dict[int, AUResult] = field(default_factory=dict)  # AU分析结果 {face_id: AUResult}
    hidden_emotions: Dict[int, HiddenEmotionResult] = field(default_factory=dict)  # 隐藏情绪结果
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def __post_init__(self):
        """初始化后处理，根据图像数据更新维度信息（如果未提供或不一致）"""
        if self.image is not None:
            # 首先获取图像的实际维度
            img_height, img_width = self.image.shape[:2]
            
            # 仅当通过构造函数传入的width/height为0 (表示未指定)
            # 或者传入的width/height与图像实际维度不符时，才用图像的实际维度更新它们
            if self.width == 0 or self.width != img_width:
                self.width = img_width
            if self.height == 0 or self.height != img_height:
                self.height = img_height
            
            # 更新通道数
            if len(self.image.shape) == 3:
                self.channels = self.image.shape[2]
            elif len(self.image.shape) == 2: # 明确处理灰度图情况
                self.channels = 1
            # 如果图像维度既不是2也不是3（异常情况），self.channels 将保留其构造时传入的值或默认值0
        # 如果 self.image is None, width, height, channels 将保留其构造时传入的值或默认值(0,0,0)
    
    def add_face(self, face: FaceDetection) -> None:
        """添加人脸检测结果"""
        self.faces[face.face_id] = face
    
    def add_emotion(self, face_id: int, emotion: EmotionResult) -> None:
        """添加情绪分析结果"""
        self.emotions[face_id] = emotion
    
    def add_au_result(self, face_id: int, au_result: AUResult) -> None:
        """添加AU分析结果"""
        self.aus[face_id] = au_result
    
    def add_hidden_emotion(self, face_id: int, hidden_emotion: HiddenEmotionResult) -> None:
        """添加隐藏情绪结果"""
        self.hidden_emotions[face_id] = hidden_emotion
    
    def get_face(self, face_id: int) -> Optional[FaceDetection]:
        """获取指定ID的人脸检测结果"""
        return self.faces.get(face_id)
    
    def get_emotion(self, face_id: int) -> Optional[EmotionResult]:
        """获取指定ID的情绪分析结果"""
        return self.emotions.get(face_id)
    
    def get_au_result(self, face_id: int) -> Optional[AUResult]:
        """获取指定ID的AU分析结果"""
        return self.aus.get(face_id)
    
    def get_hidden_emotion(self, face_id: int) -> Optional[HiddenEmotionResult]:
        """获取指定ID的隐藏情绪结果"""
        return self.hidden_emotions.get(face_id)
    
    def has_faces(self) -> bool:
        """是否检测到人脸"""
        return len(self.faces) > 0
    
    def get_dominant_face(self) -> Optional[FaceDetection]:
        """获取主要人脸（通常是最大的）"""
        if not self.faces:
            return None
        
        # 按面积大小排序，返回最大的
        return max(self.faces.values(), key=lambda face: face.face_box.area)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
    
    def copy_without_image(self) -> 'Frame':
        """创建不包含图像数据的副本（用于传输和存储）"""
        frame_copy = Frame(
            frame_id=self.frame_id,
            image=None,
            timestamp=self.timestamp,
            width=self.width,
            height=self.height,
            channels=self.channels
        )
        frame_copy.faces = self.faces.copy()
        frame_copy.emotions = self.emotions.copy()
        frame_copy.aus = self.aus.copy()
        frame_copy.hidden_emotions = self.hidden_emotions.copy()
        frame_copy.metadata = self.metadata.copy()
        return frame_copy 