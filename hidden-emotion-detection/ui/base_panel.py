#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - UI面板基类
为所有面板提供基础功能和一致的视觉风格
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
from PIL import Image, ImageDraw, ImageFont
import unicodedata # 导入 unicodedata

from ..core.data_types import FrameResult

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BasePanel")

class BasePanel:
    """UI面板基类，提供所有面板共用的基础功能"""
    
    def __init__(self, title: str):
        """
        初始化面板
        
        Args:
            title: 面板标题
        """
        self.title = title
        self.visible = True
        self.collapsed = False
        
        # 面板默认颜色和样式
        self.title_bar_color = (40, 40, 45)  # 标题栏背景色
        self.title_text_color = (230, 230, 230)  # 标题文字颜色
        self.panel_bg_color = (25, 25, 30)  # 面板背景色
        self.panel_border_color = (60, 60, 65)  # 面板边框颜色
        self.panel_shadow_color = (15, 15, 20)  # 面板阴影颜色
        
        # 标题栏高度
        self.title_bar_height = 30
        
        # 面板内边距
        self.padding = 10
        
        # 字体
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 中文字体设置
        self.cn_font_path = None
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {} # 添加字体缓存
        # 尝试加载中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "./fonts/simhei.ttf",           # 程序目录下的黑体
            "./hidden/fonts/simhei.ttf",    # hidden/fonts目录下的黑体
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Linux字体
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                self.cn_font_path = path
                break
        
        if not self.cn_font_path:
            logger.warning(f"面板 '{title}' 未找到中文字体，将使用ASCII替代")
    
    def update(self, result: FrameResult = None):
        """
        更新面板内容
        
        Args:
            result: 帧分析结果
        """
        # 基类中为空实现，由子类重写
        pass
        
    def toggle_visibility(self):
        """切换面板可见性"""
        self.visible = not self.visible
        
    def toggle_collapse(self):
        """切换面板折叠状态"""
        self.collapsed = not self.collapsed
        
    def draw_panel_frame(self, canvas: np.ndarray, x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        绘制面板框架，包括标题栏、边框和阴影
        
        Args:
            canvas: 要渲染的画布
            x: 面板左上角x坐标
            y: 面板左上角y坐标
            width: 面板宽度
            height: 面板高度
            
        Returns:
            内容区域的(x, y, width, height)元组
        """
        # 如果面板不可见，直接返回
        if not self.visible:
            return (x, y + self.title_bar_height, width, 0)
            
        # 绘制阴影
        shadow_offset = 5
        shadow_x = x + shadow_offset
        shadow_y = y + shadow_offset
        cv2.rectangle(canvas, 
                     (shadow_x, shadow_y), 
                     (shadow_x + width, shadow_y + height), 
                     self.panel_shadow_color, -1)
        
        # 绘制面板背景
        cv2.rectangle(canvas, 
                     (x, y), 
                     (x + width, y + height), 
                     self.panel_bg_color, -1)
        
        # 绘制面板边框
        cv2.rectangle(canvas, 
                     (x, y), 
                     (x + width, y + height), 
                     self.panel_border_color, 1)
        
        # 绘制标题栏
        cv2.rectangle(canvas, 
                     (x, y), 
                     (x + width, y + self.title_bar_height), 
                     self.title_bar_color, -1)
        
        # 绘制标题文本
        self.put_text(canvas, self.title, 
                     (x + 10, y + self.title_bar_height - 10), 
                     self.title_text_color, 16)
        
        # 如果面板已折叠，则只显示标题栏
        if self.collapsed:
            return (x, y + self.title_bar_height, width, 0)
        
        # 计算内容区域
        content_x = x + self.padding
        content_y = y + self.title_bar_height + self.padding
        content_width = width - 2 * self.padding
        content_height = height - self.title_bar_height - 2 * self.padding
        
        return (content_x, content_y, content_width, content_height)
        
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上
        
        Args:
            canvas: 要渲染的画布
            x: 面板左上角x坐标
            y: 面板左上角y坐标
            width: 面板宽度
            height: 面板高度
        """
        # 基类中仅绘制框架，内容由子类实现
        self.draw_panel_frame(canvas, x, y, width, height)
        
    def put_text(self, canvas: np.ndarray, text: str, position: Tuple[int, int], 
                color: Tuple[int, int, int], font_size: int, thickness: int = 1):
        """
        在画布上绘制文本，支持中文，并优化性能
        
        Args:
            canvas: 要渲染的画布
            text: 要绘制的文本
            position: 文本位置(x, y)
            color: 文本颜色(B, G, R)
            font_size: 字体大小 (像素)
            thickness: 线条粗细 (仅对cv2.putText有效)
        """
        # 检查是否只包含ASCII字符
        is_ascii = all(ord(c) < 128 for c in text)
        
        if is_ascii or self.cn_font_path is None:
            # 如果是纯ASCII或没有中文字体，使用OpenCV绘制
            # 调整 fontScale - 这需要根据字体和期望大小进行调整，0.5 是一个起点
            font_scale = font_size / 25.0 # 尝试用一个比例因子
            try:
                # (x, y) 是文本左下角的位置
                cv2.putText(canvas, text, position, self.font, font_scale, color, thickness, cv2.LINE_AA)
            except Exception as e:
                 logger.error(f"cv2.putText 绘制失败: {e}")
                 # 可以添加备用方案，例如不绘制
            return
            
        # --- 处理包含非ASCII字符的情况 (优化版) ---
        try:
            # 1. 从缓存获取或加载字体
            font_key = (self.cn_font_path, font_size)
            if font_key not in self.font_cache:
                logger.debug(f"缓存字体: {font_key}")
                self.font_cache[font_key] = ImageFont.truetype(self.cn_font_path, font_size)
            font = self.font_cache[font_key]

            # 2. 计算文本边界框 (使用 getbbox for PIL >= 9.2.0 or textbbox)
            # textbbox is more accurate, getbbox might overestimate slightly
            try:
                 # Prefer textbbox if available (PIL >= 8.0.0)
                 bbox = font.getbbox(text) # (left, top, right, bottom)
                 text_width = bbox[2] - bbox[0]
                 text_height = bbox[3] - bbox[1]
                 # Adjust position based on bbox[1] (ascent)
                 draw_y = position[1] - bbox[3] # Use bottom from bbox
            except AttributeError:
                 # Fallback for older PIL versions
                 text_width, text_height = font.getsize(text) 
                 # Fallback adjustment, might be less accurate
                 draw_y = position[1] - text_height 

            # 3. 定义目标区域 (Region of Interest - ROI) 在主画布上
            x, y = position[0], max(0, draw_y) # 确保 y 不为负
            roi_x1 = x
            roi_y1 = y
            # Add some padding to width/height to avoid clipping? Maybe 2 pixels? 
            padding_w = 2
            padding_h = 2
            roi_x2 = x + text_width + padding_w
            roi_y2 = y + text_height + padding_h

            # 确保 ROI 在画布范围内
            canvas_h, canvas_w = canvas.shape[:2]
            roi_x1 = max(0, roi_x1)
            roi_y1 = max(0, roi_y1)
            roi_x2 = min(canvas_w, roi_x2)
            roi_y2 = min(canvas_h, roi_y2)
            
            # 检查 ROI 是否有效 (防止宽度或高度为零或负数)
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                 logger.warning(f"计算出的文本 ROI 无效: pos={position}, size=({text_width},{text_height}), roi=({roi_x1},{roi_y1},{roi_x2},{roi_y2})")
                 return # Skip drawing if ROI is invalid
                 
            # 4. 提取 ROI
            roi = canvas[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # 5. 将 ROI 转换为 PIL 图像 (只转换小区域!)
            # Important: Make a copy if roi might be modified later, but BGR2RGB already copies
            pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_roi)
            
            # 6. 在 PIL ROI 上绘制文本 (相对于 ROI 的左上角)
            color_rgb = (color[2], color[1], color[0])
            # Draw at (0, 0) of the small PIL image
            draw.text((0, 0), text, font=font, fill=color_rgb)
            
            # 7. 将绘制好的 PIL ROI 转换回 OpenCV 格式
            updated_roi_bgr = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
            
            # 8. 将更新后的 ROI 粘贴回主画布
            canvas[roi_y1:roi_y2, roi_x1:roi_x2] = updated_roi_bgr

        except Exception as e:
            logger.error(f"绘制中文文本失败 (字体: {self.cn_font_path}): {e}")
            # 备用方案 (保持不变)
            text_ascii = ''.join([char if ord(char) < 128 else ' ' for char in text])
            font_scale = font_size / 25.0 
            cv2.putText(canvas, text_ascii, position, self.font, font_scale, color, thickness, cv2.LINE_AA)
    
    def draw_rounded_rect(self, canvas: np.ndarray, x: int, y: int, width: int, height: int, 
                          color: Tuple[int, int, int], radius: int = 10, thickness: int = -1):
        """
        绘制圆角矩形
        
        Args:
            canvas: 要渲染的画布
            x: 左上角x坐标
            y: 左上角y坐标
            width: 宽度
            height: 高度
            color: 颜色
            radius: 圆角半径
            thickness: 线条粗细，-1为填充
        """
        # 绘制主矩形
        cv2.rectangle(canvas, 
                     (x + radius, y), 
                     (x + width - radius, y + height), 
                     color, thickness)
        cv2.rectangle(canvas, 
                     (x, y + radius), 
                     (x + width, y + height - radius), 
                     color, thickness)
        
        # 绘制四个圆角
        cv2.circle(canvas, (x + radius, y + radius), radius, color, thickness)
        cv2.circle(canvas, (x + width - radius, y + radius), radius, color, thickness)
        cv2.circle(canvas, (x + radius, y + height - radius), radius, color, thickness)
        cv2.circle(canvas, (x + width - radius, y + height - radius), radius, color, thickness)
    
    def draw_progress_bar(self, canvas: np.ndarray, x: int, y: int, width: int, height: int, 
                         value: float, min_val: float = 0.0, max_val: float = 1.0, 
                         bg_color: Tuple[int, int, int] = (60, 60, 65), 
                         fill_color: Tuple[int, int, int] = (0, 165, 255)):
        """
        绘制进度条
        
        Args:
            canvas: 要渲染的画布
            x: 左上角x坐标
            y: 左上角y坐标
            width: 宽度
            height: 高度
            value: 当前值
            min_val: 最小值
            max_val: 最大值
            bg_color: 背景颜色
            fill_color: 填充颜色
        """
        # 绘制背景
        cv2.rectangle(canvas, (x, y), (x + width, y + height), bg_color, -1)
        
        # 计算填充宽度
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_value = max(0.0, min(1.0, normalized_value))  # 限制在0-1范围内
        fill_width = int(normalized_value * width)
        
        # 绘制填充部分
        if fill_width > 0:
            cv2.rectangle(canvas, (x, y), (x + fill_width, y + height), fill_color, -1)
    
    def draw_horizontal_separator(self, canvas: np.ndarray, x: int, y: int, width: int, 
                                 color: Tuple[int, int, int] = (60, 60, 65), thickness: int = 1):
        """
        绘制水平分隔线
        
        Args:
            canvas: 要渲染的画布
            x: 起点x坐标
            y: 起点y坐标
            width: 线宽
            color: 线条颜色
            thickness: 线条粗细
        """
        cv2.line(canvas, (x, y), (x + width, y), color, thickness) 