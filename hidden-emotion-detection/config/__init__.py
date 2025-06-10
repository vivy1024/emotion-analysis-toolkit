#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模块
负责系统配置和模型路径管理
"""

from .config_manager import ConfigManager
# from .models import ModelsConfig # 不再在此处导入和创建 ModelsConfig

# 创建全局配置管理器实例
config_manager = ConfigManager("enhance_hidden/config/config.json")

# 创建全局模型配置实例
# models_config = ModelsConfig() # 移除 ModelsConfig 的创建 