#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版界面布局管理器
为增强版隐藏情绪检测系统提供七列式可视化界面
使用独立面板组件实现
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

# 导入各个独立面板组件
from .base_panel import BasePanel
from .video_panel import VideoPanel
from .face_panel import FacePanel

from .au_intensity_panel import AUIntensityPanel
from .macro_emotion_panel import MacroEmotionPanel
from .micro_emotion_panel import MicroEmotionPanel
from .hidden_emotion_panel import HiddenEmotionPanel

from ..core.data_types import (
    EmotionType,
    FaceDetection,
    EmotionResult,
    AUResult,
    HiddenEmotionResult,
    FrameResult
)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LayoutManager")

class EnhancedUILayout:
    """增强版界面布局管理器，用于六列式可视化显示情绪分析结果"""
    
    def __init__(self, event_bus, window_name="增强版隐藏情绪检测系统", window_width=1920, window_height=1080):
        """
        初始化六列式界面布局
        
        Args:
            event_bus: 事件总线实例
            window_name: 窗口名称
            window_width: 窗口宽度
            window_height: 窗口高度
        """
        self.event_bus = event_bus # 存储事件总线
        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height
        self.canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # 面板尺寸和位置配置
        self.panel_margin = 4  # 减小边距
        self.header_height = 30
        
        # 七列布局设计
        # 1列: 实时视频
        # 2列: 人脸提取+姿态
        # 3列: AU综合状态/强度 (合并后)
        # 4列: 宏观表情
        # 5列: 微表情
        # 6列: 隐藏情绪
        
        # 计算列宽 (总共7个边距，因为现在是6列)
        total_margin = self.panel_margin * 7 
        
        # 重新定义列宽比例 (6列)
        # 视频(1), 人脸(2), AU综合(3), 宏(4), 微(5), 隐藏(6)
        # Increase width significantly for VideoPanel (index 0)
        # Reset Face (1) and AU (2) to normal/slightly larger, adjust others
        # Reduce FacePanel width slightly as it has less vertical content now
        col_ratios = [1.8, 0.9, 1.1, 0.7, 0.7, 0.8] # Video=1.8, Face=0.9, Hidden=0.8
        total_ratio = sum(col_ratios)
        available_width = window_width - total_margin
        
        self.column_widths = [int(available_width * ratio / total_ratio) for ratio in col_ratios]
        
        # 各列起始x坐标
        self.columns_x = []
        x_pos = self.panel_margin
        for width in self.column_widths:
            self.columns_x.append(x_pos)
            x_pos += width + self.panel_margin
        
        # 面板高度配置
        self.panel_height = window_height - self.panel_margin * 2
        
        # 创建各个面板实例
        # 需要 event_bus 的面板 - 将 Macro 和 Micro 添加进来
        event_bus_panels = {
            MacroEmotionPanel, # <<< 添加
            MicroEmotionPanel, # <<< 添加
            HiddenEmotionPanel, 
            AUIntensityPanel, # AU 面板现在也通过事件更新
            FacePanel,        # 人脸面板也可能需要事件来更新特定信息
            # 其他需要事件的面板...
        }
        panel_kwargs = {'event_bus': self.event_bus}

        # 使用 panel_kwargs 创建需要 event_bus 的面板
        self.video_panel = VideoPanel() # 视频面板通常不需要 event_bus
        self.face_panel = FacePanel(**panel_kwargs) if FacePanel in event_bus_panels else FacePanel()
        self.au_intensity_panel = AUIntensityPanel(**panel_kwargs) if AUIntensityPanel in event_bus_panels else AUIntensityPanel()
        self.macro_emotion_panel = MacroEmotionPanel(**panel_kwargs) if MacroEmotionPanel in event_bus_panels else MacroEmotionPanel()
        self.micro_emotion_panel = MicroEmotionPanel(**panel_kwargs) if MicroEmotionPanel in event_bus_panels else MicroEmotionPanel()
        self.hidden_emotion_panel = HiddenEmotionPanel(**panel_kwargs) # 确认需要 event_bus
        
        # 添加面板到列表（保持顺序）
        self.panels = [
            self.video_panel,
            self.face_panel,
            self.au_intensity_panel,
            self.macro_emotion_panel,
            self.micro_emotion_panel,
            self.hidden_emotion_panel
        ]
        
        # 记录FPS和时间
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        
        logger.info("七列式界面布局管理器初始化完成")
    
    def clear_canvas(self):
        """清空画布"""
        self.canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            
    def update_fps(self, fps=None):
        """更新FPS计数"""
        if fps is not None:
            self.fps = fps
        else:
            self.frame_count += 1
            elapsed = time.time() - self.last_fps_update
            
            # 每秒更新一次FPS
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_update = time.time()
    
    def update(self, frame, result=None, fps=None):
        """
        更新所有面板
        
        Args:
            frame: 输入视频帧
            result: 帧分析结果
            fps: 帧率
        """
        # 清空画布
        self.clear_canvas()
        
        # 如果没有结果，创建一个空的
        if result is None:
            result = FrameResult(frame_id=self.frame_count)
        
        # 确保结果对象中有原始帧数据
        result.frame = frame
        
        # 更新FPS
        self.update_fps(fps)
        
        # 更新视频面板
        self.video_panel.update(frame, result, self.fps)
        
        # 更新其他面板
        self.face_panel.update(frame)
        self.macro_emotion_panel.update(result)
        self.micro_emotion_panel.update(result)
        self.hidden_emotion_panel.update(result)
        
        # 更新帧计数
        self.frame_count += 1
    
    def render(self):
        """渲染所有面板到画布上"""
        # 依次渲染每个面板
        for i, panel in enumerate(self.panels):
            panel.render(
                self.canvas, 
                self.columns_x[i], 
                self.panel_margin, 
                self.column_widths[i],
                self.panel_height
            )
    
    def show(self):
        """显示界面"""
        # 先渲染所有面板
        self.render()
        
        # 显示画布
        cv2.imshow(self.window_name, self.canvas)
    
    def handle_events(self):
        """处理键盘事件"""
        key = cv2.waitKey(1) & 0xFF
        
        # ESC键退出
        if key == 27:
            return False
            
        return True 