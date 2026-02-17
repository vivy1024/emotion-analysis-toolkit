#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI设置模块
管理用户界面的各种配置参数
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Callable

from PyQt5.QtCore import QObject, pyqtSignal

from .config_manager import ConfigManager
from hidden_emotion_detection.config import config_manager # 导入全局配置实例

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UISettings")

class UISettings(QObject):
    """UI设置类，管理界面配置参数"""
    
    _instance = None  # 单例实例
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UISettings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化UI设置"""
        super().__init__()
        
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
            
        # 获取全局配置实例
        # self.config_manager = ConfigManager() # 不再自行初始化
        self.config_manager = config_manager
        
        # 加载初始设置
        self._load_settings()
        
        # 注册配置变更回调
        self.config_manager.register_change_callback("ui", self._on_ui_config_changed)
        
        self._initialized = True
        logger.info("UI设置初始化完成")
    
    def _on_ui_config_changed(self, path: str, value: Any):
        """UI配置变更回调"""
        logger.debug(f"UI配置已更改: {path} = {value}")
        self._load_settings()
    
    def _load_settings(self):
        """重新加载UI设置"""
        # 主题设置
        self.theme = self.config_manager.get("ui.theme", "dark")
        
        # 显示选项
        self.show_debugging = self.config_manager.get("ui.show_debugging", False)
        self.show_confidence = self.config_manager.get("ui.show_confidence", True)
        self.show_au_details = self.config_manager.get("ui.show_au_details", False)
        self.show_charts = self.config_manager.get("ui.show_charts", True)
        
        # 图表设置
        self.chart_history_length = self.config_manager.get("ui.chart_history_length", 50)
        
        # 布局设置
        self.fullscreen = self.config_manager.get("ui.fullscreen", False)
        self.display_mode = self.config_manager.get("ui.display_mode", "combined")
        
        # 颜色主题
        self.setup_color_theme()
        
        logger.debug("UI设置已重新加载")
    
    def setup_color_theme(self):
        """设置颜色主题"""
        # 暗色主题
        dark_theme = {
            "background": "#2D2D30",
            "panel_bg": "#252526",
            "text": "#E8E8E8",
            "text_secondary": "#B0B0B0",
            "border": "#3F3F46",
            "accent": "#007ACC",
            "chart_colors": ["#4CAF50", "#F44336", "#3F51B5", "#FFC107", "#9C27B0", "#FF5722", "#2196F3"],
            "emotion_colors": {
                "neutral": "#9E9E9E",
                "happiness": "#4CAF50",
                "sadness": "#3F51B5",
                "surprise": "#FFC107",
                "fear": "#9C27B0",
                "anger": "#F44336",
                "disgust": "#FF5722",
                "contempt": "#795548"
            }
        }
        
        # 亮色主题
        light_theme = {
            "background": "#F8F8F8",
            "panel_bg": "#FFFFFF",
            "text": "#212121",
            "text_secondary": "#757575",
            "border": "#E0E0E0",
            "accent": "#2196F3",
            "chart_colors": ["#388E3C", "#D32F2F", "#303F9F", "#FFA000", "#7B1FA2", "#E64A19", "#1976D2"],
            "emotion_colors": {
                "neutral": "#616161",
                "happiness": "#388E3C",
                "sadness": "#303F9F",
                "surprise": "#FFA000",
                "fear": "#7B1FA2",
                "anger": "#D32F2F",
                "disgust": "#E64A19",
                "contempt": "#5D4037"
            }
        }
        
        # 根据当前主题选择颜色
        self.colors = dark_theme if self.theme == "dark" else light_theme
    
    def get_color(self, name: str) -> str:
        """
        获取颜色
        
        Args:
            name: 颜色名称，如'background', 'text'等
            
        Returns:
            str: 颜色代码
        """
        return self.colors.get(name)
    
    def get_emotion_color(self, emotion_name: str) -> str:
        """
        获取情绪颜色
        
        Args:
            emotion_name: 情绪名称，如'happiness', 'sadness'等
            
        Returns:
            str: 颜色代码
        """
        return self.colors.get("emotion_colors", {}).get(emotion_name.lower(), "#808080")
    
    def get_chart_color(self, index: int) -> str:
        """
        获取图表颜色
        
        Args:
            index: 颜色索引
            
        Returns:
            str: 颜色代码
        """
        chart_colors = self.colors.get("chart_colors", [])
        if not chart_colors:
            return "#808080"
        
        return chart_colors[index % len(chart_colors)]
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        更新单个设置
        
        Args:
            key: 设置键名
            value: 设置值
            
        Returns:
            bool: 是否成功更新
        """
        config_path = f"ui.{key}"
        success = self.config_manager.set(config_path, value)
        
        if success:
            # 更新本地缓存
            setattr(self, key, value)
            
            # 如果更新的是主题，重新加载颜色
            if key == "theme":
                self.setup_color_theme()
                
            logger.debug(f"UI设置已更新: {key} = {value}")
            
        return success
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        获取所有UI设置
        
        Returns:
            Dict[str, Any]: 所有UI设置
        """
        return {
            "theme": self.theme,
            "show_debugging": self.show_debugging,
            "show_confidence": self.show_confidence,
            "show_au_details": self.show_au_details,
            "show_charts": self.show_charts,
            "chart_history_length": self.chart_history_length,
            "fullscreen": self.fullscreen,
            "display_mode": self.display_mode,
            "colors": self.colors
        }

# 创建全局实例
ui_settings = UISettings() 