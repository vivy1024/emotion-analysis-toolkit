#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏表情检测系统 - 用户界面模块
"""

try:
    from .layout_manager import EnhancedUILayout
except ImportError as e:
    import logging
    logging.warning("布局管理器导入失败")
    logging.error(f"导入失败: {e}")
    EnhancedUILayout = None

try:
    from .video_panel import VideoPanel
except ImportError:
    import logging
    logging.warning("视频面板导入失败")
    VideoPanel = None

try:
    from .face_panel import FacePanel
except ImportError:
    import logging
    logging.warning("人脸面板导入失败")
    FacePanel = None

try:
    from .macro_emotion_panel import MacroEmotionPanel
except ImportError:
    import logging
    logging.warning("宏观情绪面板导入失败")
    MacroEmotionPanel = None

try:
    from .micro_emotion_panel import MicroEmotionPanel
except ImportError:
    import logging
    logging.warning("微表情面板导入失败")
    MicroEmotionPanel = None

try:
    # logging.warning("AU状态面板导入失败")
    from .au_state_panel import AUStatePanel
except ImportError:
    AUStatePanel = None

try:
    from .au_intensity_panel import AUIntensityPanel
except ImportError:
    import logging
    logging.warning("AU强度面板导入失败")
    AUIntensityPanel = None

try:
    from .hidden_emotion_panel import HiddenEmotionPanel
except ImportError:
    import logging
    logging.warning("隐藏情绪面板导入失败")
    HiddenEmotionPanel = None

__all__ = [
    'EnhancedUILayout',
    'VideoPanel',
    'FacePanel',
    'MacroEmotionPanel',
    'MicroEmotionPanel',
    'AUStatePanel',
    'AUIntensityPanel',
    'HiddenEmotionPanel'
] 