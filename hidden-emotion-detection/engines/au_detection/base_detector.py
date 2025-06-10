#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BaseAUDetector基类，定义AU检测接口
此文件只是作为接口定义，实际实现在sklearn_au_detector.py中
"""

import logging

class BaseAUDetector:
    """基础AU检测器接口定义"""
    
    def __init__(self, **kwargs):
        """初始化基础检测器"""
        self.logger = logging.getLogger("AUDetector")
        self.logger.info("初始化基础AU检测器")
        
    def detect_face_aus(self, face_image):
        """
        从人脸图像检测AU
        
        参数:
            face_image: 人脸图像
            
        返回:
            包含AU检测结果的字典: {'aus': {AU_ID: intensity}, 'au_present': {AU_ID: is_present}}
        """
        return {'aus': {}, 'au_present': {}}
        
    def detect_from_landmarks(self, landmarks):
        """
        从landmarks检测AU (兼容性方法)
        
        参数:
            landmarks: 面部关键点
            
        返回:
            包含AU检测结果的字典
        """
        return self.detect_face_aus(None)
        
    def detect_from_frame(self, frame):
        """
        从完整帧检测AU (兼容性方法)
        
        参数:
            frame: 完整图像帧
            
        返回:
            包含AU检测结果的字典
        """
        return self.detect_face_aus(None)