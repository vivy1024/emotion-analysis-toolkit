#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型配置模块
统一管理所有模型路径和配置
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .config_manager import ConfigManager
from hidden_emotion_detection.config import config_manager # 重新导入全局实例

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelsConfig")

class ModelsConfig:
    """模型配置类，管理所有模型路径和配置"""
    
    def __init__(self, config_manager: ConfigManager):
        """初始化模型配置，接收 ConfigManager 实例作为参数"""
        # 使用传入的配置管理器实例
        self.config_manager = config_manager 
        
        # 存储模型文件路径
        self.model_paths = {}
        
        # 更新模型路径
        self.update_model_paths()
        
        # 注册配置变更回调
        self.config_manager.register_change_callback("system.models_dir", self._on_models_dir_changed)
        self.config_manager.register_change_callback("macro.model_path", lambda p, v: self.update_model_paths())
        self.config_manager.register_change_callback("micro.model_path", lambda p, v: self.update_model_paths())
        self.config_manager.register_change_callback("au.models_dir", lambda p, v: self.update_model_paths())
        self.config_manager.register_change_callback("face.landmark_model", lambda p, v: self.update_model_paths())
        
        logger.info("模型配置初始化完成")
    
    def _on_models_dir_changed(self, path: str, value: str):
        """模型目录变更回调"""
        logger.info(f"模型目录已更改: {value}")
        self.update_model_paths()
    
    def update_model_paths(self):
        """更新所有模型路径"""
        # 获取项目根目录和模型根目录
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.MODELS_ROOT = self.config_manager.get("system.models_dir")
        
        # 确保模型根目录是绝对路径
        if not os.path.isabs(self.MODELS_ROOT):
            self.MODELS_ROOT = os.path.join(self.PROJECT_ROOT, self.MODELS_ROOT)
        
        # 特征点模型
        landmark_model_filename = self.config_manager.get("face.landmark_model", "shape_predictor_68_face_landmarks.dat")
        self.LANDMARK_MODEL = os.path.join(self.MODELS_ROOT, landmark_model_filename)
        
        # AU模型目录
        self.AU_MODELS_DIR = self.config_manager.get("au.models_dir")
        if not os.path.isabs(self.AU_MODELS_DIR):
            self.AU_MODELS_DIR = os.path.join(self.PROJECT_ROOT, self.AU_MODELS_DIR)
        
        self.AU_MODELS_INDEX = os.path.join(self.AU_MODELS_DIR, 
                                          self.config_manager.get("au.au_model_index", "AU_all_static.txt"))
        self.AU_MODELS_SVM = os.path.join(self.AU_MODELS_DIR, "svm_combined")
        
        # 情绪分析模型
        self.MACRO_EMOTION_MODEL = self.config_manager.get("macro.model_path")
        if not os.path.isabs(self.MACRO_EMOTION_MODEL):
            self.MACRO_EMOTION_MODEL = os.path.join(self.PROJECT_ROOT, self.MACRO_EMOTION_MODEL)
        
        self.MICRO_EMOTION_MODEL = self.config_manager.get("micro.model_path")
        if not os.path.isabs(self.MICRO_EMOTION_MODEL):
            self.MICRO_EMOTION_MODEL = os.path.join(self.PROJECT_ROOT, self.MICRO_EMOTION_MODEL)
        
        # 检查模型是否存在
        self.missing_models = self.check_models_exist()
        
        if self.missing_models:
            logger.warning(f"发现{len(self.missing_models)}个缺失的模型文件")
        else:
            logger.info("所有模型文件检查通过")
    
    def check_models_exist(self) -> List[Tuple[str, str]]:
        """
        检查关键模型文件是否存在
        
        Returns:
            List[Tuple[str, str]]: 缺失的模型列表，元素为(模型名称, 模型路径)
        """
        missing_models = []
        
        # 检查人脸特征点模型
        if not os.path.exists(self.LANDMARK_MODEL):
            missing_models.append(("人脸特征点模型", self.LANDMARK_MODEL))
            logger.warning(f"人脸特征点模型不存在: {self.LANDMARK_MODEL}")
        
        # 检查AU模型索引
        if not os.path.exists(self.AU_MODELS_INDEX):
            missing_models.append(("AU模型索引", self.AU_MODELS_INDEX))
            logger.warning(f"AU模型索引不存在: {self.AU_MODELS_INDEX}")
        
        # 检查AU模型目录
        if not os.path.exists(self.AU_MODELS_SVM):
            missing_models.append(("AU模型目录", self.AU_MODELS_SVM))
            logger.warning(f"AU模型目录不存在: {self.AU_MODELS_SVM}")
        
        # 检查宏观情绪模型
        if not os.path.exists(self.MACRO_EMOTION_MODEL):
            missing_models.append(("宏观情绪模型", self.MACRO_EMOTION_MODEL))
            logger.warning(f"宏观情绪模型不存在: {self.MACRO_EMOTION_MODEL}")
        
        # 检查微表情模型
        if not os.path.exists(self.MICRO_EMOTION_MODEL):
            missing_models.append(("微表情模型", self.MICRO_EMOTION_MODEL))
            logger.warning(f"微表情模型不存在: {self.MICRO_EMOTION_MODEL}")
        
        return missing_models
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        获取指定模型的路径
        
        Args:
            model_name: 模型名称，如'landmark', 'macro', 'micro', 'au_index', 'au_svm'
            
        Returns:
            Optional[str]: 模型路径，如果模型不存在则返回None
        """
        model_paths = {
            'landmark': self.LANDMARK_MODEL,
            'macro': self.MACRO_EMOTION_MODEL,
            'micro': self.MICRO_EMOTION_MODEL,
            'au_index': self.AU_MODELS_INDEX,
            'au_svm': self.AU_MODELS_SVM
        }
        
        if model_name not in model_paths:
            logger.warning(f"未知的模型名称: {model_name}")
            return None
        
        model_path = model_paths[model_name]
        if not os.path.exists(model_path):
            logger.warning(f"模型路径不存在: {model_path}")
            return None
        
        return model_path

# AU数据结构转换函数
def convert_au_data(aus, format_type="ui"):
    """
    转换AU数据结构，适配不同模块间的数据交换
    
    Args:
        aus: AU数据，可以是多种格式
        format_type: 目标格式类型，'ui'或'engine'
    
    Returns:
        转换后的AU数据
    """
    if format_type == "ui":
        # 转换为UI格式（"1": True, "2": 0.8 等）
        if isinstance(aus, dict) and any("AU" in k for k in aus.keys()):
            # 从AU01格式转换
            au_present = {}
            au_intensities = {}
            
            for au_id, au_data in aus.items():
                # 去除"AU"前缀，只保留数字
                id_num = au_id.replace("AU", "").lstrip("0")
                
                if isinstance(au_data, dict):
                    # {"AU01": {"name": "内眉上扬", "intensity": 0.8, "present": True}}
                    au_present[id_num] = au_data.get("present", False)
                    au_intensities[id_num] = au_data.get("intensity", 0.0)
                else:
                    # 其他格式处理
                    au_present[id_num] = bool(au_data > 0.5 if isinstance(au_data, (int, float)) else au_data)
                    au_intensities[id_num] = float(au_data if isinstance(au_data, (int, float)) else 0.0)
            
            return au_intensities, au_present
        
        elif isinstance(aus, tuple) and len(aus) == 2:
            # 已经是UI格式，返回原样
            return aus
        
        else:
            logger.warning(f"无法识别的AU数据格式: {type(aus)}")
            return {}, {}
    
    elif format_type == "engine":
        # 转换为引擎格式 ({"AU01": {...}})
        au_data = {}
        
        if isinstance(aus, tuple) and len(aus) == 2:
            # 从UI格式转换
            au_intensities, au_present = aus
            
            for au_id in set(list(au_intensities.keys()) + list(au_present.keys())):
                # 添加AU前缀，确保两位数格式
                engine_id = f"AU{int(au_id):02d}"
                
                au_data[engine_id] = {
                    "intensity": au_intensities.get(au_id, 0.0),
                    "present": au_present.get(au_id, False)
                }
        
        return au_data
    
    else:
        logger.warning(f"未知的格式类型: {format_type}")
        return aus 