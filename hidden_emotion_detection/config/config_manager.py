#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器模块
提供集中化的配置管理，支持配置文件加载、保存、热更新和验证
"""

import os
import json
import yaml
import logging
import threading
from typing import Dict, Any, Optional, List, Set, Callable
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConfigManager")

class ConfigManager:
    """
    配置管理器，负责系统配置的加载、保存和访问
    """
    
    _instance = None  # 单例实例
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 系统级配置
        "system": {
            "debug_mode": False,
            "log_level": "INFO",
            "data_dir": "data",
            "models_dir": "enhance_hidden/models",
            "openface_dir": "D:/pycharm2/PythonProject2/18/OpenFace_2.2.0_win_x64",
            "gpu_enabled": True,
            "multithreading": True,
            "thread_pool_size": 4,
            "video_resolution": (640, 480),
            "video_source": 0,
            "frame_rate": 30
        },
        
        # 人脸检测配置
        "face": {
            "detection_method": "dnn",  # dnn, haar, dlib
            "tracking_enabled": True,
            "min_face_size": 50,
            "detection_interval": 5,    # 每隔几帧进行一次检测
            "confidence_threshold": 0.6,
            "use_landmarks": True,
            "landmark_method": "dlib",  # dlib, mediapipe
            "landmark_model": "shape_predictor_68_face_landmarks.dat",
            "align_faces": True
        },
        
        # 宏观表情配置
        "macro": {
            "enabled": True,
            "model_path": "enhance_hidden/models/macro.pt",
            "model_type": "EmotionResNet",
            "threshold": 0.5,
            "gpu_acceleration": True,
            "dynamic_adjustment": True,
            "batch_size": 1
        },
        
        # 微表情配置
        "micro": {
            "enabled": True,
            "model_path": "enhance_hidden/models/micro.pt",
            "sequence_length": 20,
            "detection_interval": 10,
            "threshold": 0.6,
            "gpu_acceleration": True,
            "use_optical_flow": True
        },
        
        # AU分析配置
        "au": {
            "enabled": True,
            "models_dir": "enhance_hidden/models/AU_predictors",
            "au_model_index": "AU_all_static.txt",
            "threshold": 1.0,
            "min_intensity": 0.2,
            "use_real_detector": True,
            "gpu_acceleration": True,
            "update_interval": 5  # 每隔几帧更新一次AU分析
        },
        
        # 隐藏情绪检测配置
        "hidden": {
            "enabled": True,
            "conflict_threshold": 0.6,
            "au_weight": 0.3,
            "window_size": 10,
            "temporal_smoothing": True,
            "smoothing_factor": 0.7
        },
        
        # 用户界面配置
        "ui": {
            "theme": "dark",
            "show_debugging": False,
            "show_confidence": True,
            "show_au_details": False,
            "show_charts": True,
            "chart_history_length": 50,
            "fullscreen": False,
            "display_mode": "combined"  # combined, split, minimal
        }
    }
    
    def __new__(cls, config_path: Optional[str] = None):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，None使用默认配置
        """
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
            
        # 配置文件路径
        self.config_path = config_path
        
        # 当前配置，默认使用默认配置
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 配置修改回调函数字典，键为配置路径，值为回调函数列表
        self._change_callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
        
        # 存储已注册的回调函数和路径
        self._registered_callbacks: Dict[Callable, Set[str]] = {}
        
        # 锁，保护配置访问
        self._lock = threading.RLock()
        
        # 尝试创建配置文件目录
        if config_path:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            os.makedirs(config_dir, exist_ok=True)
            
            # 加载配置文件
            self.load_config()
        
        # 验证模型路径
        self._validate_model_paths()
        
        self._initialized = True
        logger.info("配置管理器初始化完成")
    
    def _validate_model_paths(self):
        """验证所有模型路径，如果路径无效则尝试查找正确的路径"""
        # 获取系统模型目录
        models_dir = self.get("system.models_dir", "enhance_hidden/models")
        
        # 确保模型目录存在
        if not os.path.exists(models_dir):
            models_dir = "enhance_hidden/models"
            self.set("system.models_dir", models_dir)
            logger.warning(f"模型目录不存在，使用默认路径: {models_dir}")
        
        # 检查并更新宏观表情模型路径
        macro_model_path = self.get("macro.model_path", "")
        if not os.path.exists(macro_model_path):
            new_macro_path = os.path.join(models_dir, "macro.pt")
            if os.path.exists(new_macro_path):
                self.set("macro.model_path", new_macro_path)
                logger.info(f"已更新宏观表情模型路径: {new_macro_path}")
            else:
                logger.warning(f"宏观表情模型不存在: {new_macro_path}")
        
        # 检查并更新微表情模型路径
        micro_model_path = self.get("micro.model_path", "")
        if not os.path.exists(micro_model_path):
            new_micro_path = os.path.join(models_dir, "micro.pt")
            if os.path.exists(new_micro_path):
                self.set("micro.model_path", new_micro_path)
                logger.info(f"已更新微表情模型路径: {new_micro_path}")
            else:
                logger.warning(f"微表情模型不存在: {new_micro_path}")
        
        # 检查并更新AU模型目录
        au_models_dir = self.get("au.models_dir", "")
        if not os.path.exists(au_models_dir):
            new_au_dir = os.path.join(models_dir, "AU_predictors")
            if os.path.exists(new_au_dir):
                self.set("au.models_dir", new_au_dir)
                logger.info(f"已更新AU模型目录: {new_au_dir}")
            else:
                logger.warning(f"AU模型目录不存在: {new_au_dir}")
        
        # 检查并更新特征点模型路径
        landmark_model_path = os.path.join(models_dir, self.get("face.landmark_model", "shape_predictor_68_face_landmarks.dat"))
        if not os.path.exists(landmark_model_path):
            new_landmark_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
            if os.path.exists(new_landmark_path):
                self.set("face.landmark_model", "shape_predictor_68_face_landmarks.dat")
                logger.info(f"已更新特征点模型路径: {new_landmark_path}")
            else:
                logger.warning(f"特征点模型不存在: {new_landmark_path}")
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径，None使用初始化时的路径
            
        Returns:
            bool: 是否成功加载
        """
        if config_path:
            self.config_path = config_path
            
        if not self.config_path:
            logger.warning("未指定配置文件路径，使用默认配置")
            return False
            
        # 检查文件是否存在
        if not os.path.exists(self.config_path):
            logger.warning(f"配置文件不存在: {self.config_path}，将使用默认配置")
            # 尝试创建默认配置文件
            try:
                self.save_config()
                logger.info(f"已创建默认配置文件: {self.config_path}")
            except Exception as e:
                logger.error(f"创建默认配置文件失败: {e}")
            return False
            
        try:
            # 根据文件扩展名选择加载方式
            ext = os.path.splitext(self.config_path)[1].lower()
            
            if ext == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
            elif ext in ['.yml', '.yaml']:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.error(f"不支持的配置文件格式: {ext}")
                return False
                
            # 合并配置
            with self._lock:
                self._merge_config(self.config, loaded_config)
                
            logger.info(f"成功加载配置文件: {self.config_path}")
            
            # 验证模型路径
            self._validate_model_paths()
            
            # 触发配置变更事件
            self._trigger_config_changed()
            
            return True
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            config_path: 保存路径，None使用当前路径
            
        Returns:
            bool: 是否成功保存
        """
        if config_path:
            save_path = config_path
        elif self.config_path:
            save_path = self.config_path
        else:
            logger.error("未指定配置文件保存路径")
            return False
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 根据文件扩展名选择保存方式
            ext = os.path.splitext(save_path)[1].lower()
            
            with self._lock:
                if ext == '.json':
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=4, ensure_ascii=False)
                elif ext in ['.yml', '.yaml']:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                else:
                    logger.error(f"不支持的配置文件格式: {ext}")
                    return False
                    
            logger.info(f"成功保存配置到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径，格式为"section.key"，如"system.debug_mode"
            default: 如果路径不存在，返回的默认值
            
        Returns:
            Any: 配置值
        """
        with self._lock:
            # 分割路径
            parts = path.split('.')
            
            # 从配置中获取值
            current = self.config
            try:
                for part in parts:
                    current = current[part]
                return current
            except (KeyError, TypeError):
                # 如果路径不存在，返回默认值
                return default
    
    def set(self, path: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            path: 配置路径，格式为"section.key"，如"system.debug_mode"
            value: 新的配置值
            
        Returns:
            bool: 是否成功设置
        """
        with self._lock:
            # 分割路径
            parts = path.split('.')
            
            # 逐层查找，直到最后一层
            current = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    # 如果路径不存在，创建新的字典
                    current[part] = {}
                current = current[part]
                
                # 如果当前不是字典，无法继续设置
                if not isinstance(current, dict):
                    logger.error(f"配置路径错误，'{'.'.join(parts[:i+1])}'不是字典")
                    return False
            
            # 设置值
            last_part = parts[-1]
            current[last_part] = value
            
        # 通知配置变更
        self._notify_config_changed(path, value)
        return True
    
    def register_change_callback(self, path_prefix: str, callback: Callable[[str, Any], None]):
        """
        注册配置变更回调函数
        
        Args:
            path_prefix: 配置路径前缀，如"system"
            callback: 回调函数，接收路径和新值作为参数
        """
        with self._lock:
            # 确保路径前缀的回调列表存在
            if path_prefix not in self._change_callbacks:
                self._change_callbacks[path_prefix] = []
            
            # 添加回调函数
            self._change_callbacks[path_prefix].append(callback)
            
            # 记录回调函数注册的路径
            if callback not in self._registered_callbacks:
                self._registered_callbacks[callback] = set()
            self._registered_callbacks[callback].add(path_prefix)
            
            logger.debug(f"已注册配置变更回调函数: {path_prefix} -> {callback}")
    
    def unregister_change_callback(self, callback: Callable[[str, Any], None]) -> bool:
        """
        注销配置变更回调函数
        
        Args:
            callback: 要注销的回调函数
            
        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            # 如果回调函数未注册，返回False
            if callback not in self._registered_callbacks:
                return False
            
            # 从所有路径前缀的回调列表中移除该函数
            for path_prefix in self._registered_callbacks[callback]:
                if path_prefix in self._change_callbacks:
                    if callback in self._change_callbacks[path_prefix]:
                        self._change_callbacks[path_prefix].remove(callback)
                        
                        # 如果路径前缀的回调列表为空，移除该路径前缀
                        if not self._change_callbacks[path_prefix]:
                            del self._change_callbacks[path_prefix]
            
            # 移除回调函数的注册记录
            del self._registered_callbacks[callback]
            
            logger.debug(f"已注销配置变更回调函数: {callback}")
            return True
    
    def _notify_config_changed(self, path: str, value: Any):
        """
        通知配置变更
        
        Args:
            path: 变更的配置路径
            value: 新的配置值
        """
        with self._lock:
            # 获取所有匹配的路径前缀
            matching_prefixes = []
            for prefix in self._change_callbacks:
                # 如果路径以前缀开头，或者前缀为空（匹配所有变更）
                if path.startswith(prefix) or not prefix:
                    matching_prefixes.append(prefix)
            
            # 收集所有需要调用的回调函数
            callbacks_to_call = []
            for prefix in matching_prefixes:
                callbacks_to_call.extend(self._change_callbacks[prefix])
        
        # 在锁外调用回调函数，避免死锁
        for callback in callbacks_to_call:
            try:
                callback(path, value)
            except Exception as e:
                logger.error(f"配置变更回调函数失败: {e}")
    
    def _trigger_config_changed(self):
        """触发所有配置项的变更通知"""
        # 辅助函数，递归遍历配置字典
        def traverse_config(config, prefix=""):
            for key, value in config.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    traverse_config(value, path)
                else:
                    self._notify_config_changed(path, value)
        
        # 从根配置开始遍历
        traverse_config(self.config)
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        合并配置，深度更新目标配置
        
        Args:
            target: 目标配置字典
            source: 源配置字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # 递归合并字典
                self._merge_config(target[key], value)
            else:
                # 直接更新值
                target[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            Dict[str, Any]: 所有配置
        """
        with self._lock:
            # 返回配置的深拷贝，避免外部修改
            return json.loads(json.dumps(self.config))
    
    def reset(self):
        """重置为默认配置"""
        with self._lock:
            self.config = self.DEFAULT_CONFIG.copy()
        
        # 触发配置变更事件
        self._trigger_config_changed()
        
        logger.info("已重置为默认配置") 