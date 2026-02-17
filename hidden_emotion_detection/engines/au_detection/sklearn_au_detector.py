#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SklearnAUDetector：使用py-feat SVM模型的AU检测器
此文件是对feat_svm_classifier.py的包装，仅保留py-feat的SVM模型功能
"""

import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Union

# 导入基础检测器和SklearnSVMClassifier
from hidden_emotion_detection.engines.au_detection.base_detector import BaseAUDetector
from hidden_emotion_detection.engines.au_detection.feat_svm_classifier import SklearnSVMClassifier

# 设置日志
logger = logging.getLogger("SklearnAUDetector")

class SklearnAUDetector(BaseAUDetector):
    """使用py-feat SVM模型进行AU检测"""
    
    def __init__(self, svr_models_dir="enhance_hidden/models/svm_models", **kwargs):
        """
        初始化Sklearn AU检测器
        
        Args:
            svr_models_dir: SVM模型目录，重命名为svm_models更合适
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger("SklearnAUDetector")
        self.logger.info("初始化SklearnAUDetector，使用py-feat的SVM模型")
        
        # 创建SVM分类器
        self.svm_classifier = SklearnSVMClassifier(model_dir=svr_models_dir)
        
        # AU映射表 (AU标识到名称的映射)
        self.au_names = {
            "AU01": "内眉提升",
            "AU02": "外眉提升",
            "AU04": "皱眉",
            "AU05": "上睑提升",
            "AU06": "脸颊提升",
            "AU07": "眼睑紧绷",
            "AU09": "鼻皱",
            "AU10": "上唇提升",
            "AU12": "嘴角拉伸",
            "AU14": "酒窝",
            "AU15": "嘴角下拉",
            "AU17": "下巴提升",
            "AU20": "唇部延伸",
            "AU23": "唇紧绷",
            "AU25": "嘴唇分开",
            "AU26": "下巴下降",
            "AU28": "嘴唇吸入",
            "AU45": "眨眼"
        }
        
    def detect_face_aus(self, face_image):
        """
        从人脸图像检测AU
        
        参数:
            face_image: 输入人脸图像
            
        返回:
            包含AU强度值和二元存在值的字典
        """
        self.logger.debug("使用py-feat SVM模型检测人脸AU")
        
        if face_image is None:
            self.logger.warning("输入人脸图像为空")
            return {'aus': {}, 'au_present': {}}
        
        # 使用SVM分类器获取AU检测结果
        raw_results = self.svm_classifier.detect(face_image)
        
        # 转换结果格式
        aus = {}  # 强度值 (0-1范围)
        au_present = {}  # 二元存在值
        
        # 处理结果
        for au_name, confidence in raw_results.items():
            # 将SVM的置信度转换为0-1范围的强度值
            # 通常SVM置信度是无上限的，这里做一个简单的Sigmoid映射
            intensity = 1.0 / (1.0 + np.exp(-confidence))
            
            # 标准化到0-1范围
            aus[au_name] = intensity
            
            # 置信度大于0认为AU存在
            au_present[au_name] = confidence > 0
        
        # 记录检测到的AU数量
        active_aus = sum(1 for v in au_present.values() if v)
        self.logger.info(f"检测到{active_aus}个激活的AU (总共{len(au_present)}个AU)")
        
        return {
            'aus': aus,  # 0-1范围的强度值
            'au_present': au_present  # 二元存在值
        }
        
    def detect_from_landmarks(self, landmarks):
        """
        从landmarks检测AU (保持接口兼容，但不使用landmarks)
        
        参数:
            landmarks: 面部关键点 (未使用)
            
        返回:
            空结果字典
        """
        self.logger.warning("landmarks方法被调用，但SVM检测器不支持landmarks检测")
        return {'aus': {}, 'au_present': {}}
        
    def detect_from_frame(self, frame):
        """
        从完整帧检测AU (转发到face_aus方法)
        
        参数:
            frame: 输入图像帧
            
        返回:
            包含AU检测结果的字典
        """
        self.logger.debug("frame方法被调用，转为face_aus方法")
        return self.detect_face_aus(frame) 