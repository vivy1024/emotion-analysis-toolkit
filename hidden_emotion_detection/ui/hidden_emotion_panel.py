#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 隐藏情绪面板
显示隐藏情绪分析结果
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base_panel import BasePanel
from ..core.data_types import FrameResult, EmotionType, EmotionResult, HiddenEmotionResult, EventType, Event
from ..core.event_bus import EventBus

logger = logging.getLogger(__name__)

class HiddenEmotionPanel(BasePanel):
    """显示隐藏情绪分析结果的UI面板"""

    def __init__(self, event_bus: EventBus, **kwargs):
        # 调用基类构造函数，传递标题
        super().__init__(title="隐藏情绪分析") # 确保传递 title
        # 传递其他来自 BasePanel 的 kwargs (如果需要，但当前 BasePanel 不接受)
        # super().__init__(title="隐藏情绪分析", **kwargs)
        
        self.event_bus = event_bus
        self.hidden_result: Optional[HiddenEmotionResult] = None
        self.current_face_id: Optional[int] = None # 当前关注的人脸ID

        # 定义面板特定的颜色
        self.text_color = (200, 200, 200)  # 浅灰色文本
        self.highlight_color = (100, 200, 250)  # 蓝色高亮
        self.positive_color = (100, 255, 100) # 绿色 (例如：未检测到隐藏情绪)
        self.negative_color = (50, 50, 255)   # 红色 (例如：检测到隐藏情绪)
        self.neutral_color = (180, 180, 180) # 中性灰色

        # 订阅隐藏情绪分析结果事件
        self.event_bus.subscribe(EventType.HIDDEN_EMOTION_ANALYZED, self._on_hidden_emotion_analyzed)
        logger.info("HiddenEmotionPanel 初始化完成并订阅 HIDDEN_EMOTION_ANALYZED 事件")

    def _on_hidden_emotion_analyzed(self, event: Event):
        """处理隐藏情绪分析结果事件"""
        # 使用 event.data 访问数据
        data = event.data 
        if not isinstance(data, dict):
             logger.error(f"HiddenEmotionPanel received unexpected data type in event: {type(data)}")
             return
             
        face = data.get('face')
        result = data.get('result')

        if face and result:
            # 只更新当前关注的人脸的结果
            if face.face_id == self.current_face_id:
                self.hidden_result = result
                logger.debug(f"收到并更新 Face ID {self.current_face_id} 的隐藏情绪结果: {result}")
            else:
                 logger.debug(f"收到 Face ID {face.face_id} 的隐藏情绪结果，但当前关注 {self.current_face_id}，忽略更新")
        else:
             logger.warning("收到的 HIDDEN_EMOTION_ANALYZED 事件缺少 face 或 result 数据")

    def update(self, result: FrameResult = None):
        """
        更新当前关注的人脸ID。实际结果由事件处理器更新。

        Args:
            result: 帧分析结果
        """
        new_face_id = None
        if result and result.face_detections:
            dominant_face = result.get_dominant_face()
            if dominant_face:
                new_face_id = dominant_face.face_id

        # 如果关注的人脸发生变化，清空旧结果
        if self.current_face_id != new_face_id:
             logger.debug(f"关注的人脸从 {self.current_face_id} 变为 {new_face_id}，清空当前结果")
             self.current_face_id = new_face_id
             self.hidden_result = None # 清空结果，等待新事件

    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上。先使用 BasePanel 的 render 绘制框架，再绘制动态隐藏情绪内容。
        Args:
            canvas: 要渲染的画布
            x: 面板左上角x坐标
            y: 面板左上角y坐标
            width: 面板宽度
            height: 面板高度
        """
        # 1. 调用 BasePanel 的 render 方法处理静态框架和缓冲
        try:
            super().render(canvas, x, y, width, height)
        except Exception as e:
            logger.error(f"Error during BasePanel render call from HiddenEmotionPanel: {e}")
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 50), -1)
            self.put_text(canvas, "Render Error", (x + 5, y + 20), (0, 0, 255), 14)
            return # Stop if base failed

        # 2. 检查可见性和折叠状态
        if not self.visible or self.collapsed:
            return

        # 3. 计算内容区域坐标
        title_height = 30
        padding = 5
        content_x = x + padding
        content_y = y + title_height + padding
        content_width = width - 2 * padding
        content_height = height - title_height - 2 * padding

        # 确保内容区域有效
        if content_width <= 0 or content_height <= 0:
            return
            
        # --- Start of rendering DYNAMIC hidden emotion content --- 
        center_x = content_x + content_width // 2
        
        # 添加描述文字 (绘制在内容区域)
        description = "隐藏情绪检测结果"
        desc_size, _ = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1) # For size 16 approx
        desc_x = content_x + (content_width - desc_size[0]) // 2
        desc_y = content_y + 25 # Position from top of content area
        self.put_text(canvas, description, (desc_x, desc_y), self.text_color, 16)
        
        # 绘制水平分隔线 (基于内容区域)
        separator_y = content_y + 40
        self.draw_horizontal_separator(canvas, 
                                      content_x + 10, 
                                      separator_y, 
                                      content_width - 20)
        
        # 设置动态内容的 Y 轴起始位置 (基于内容区域)
        y_offset = separator_y + 20 # Start below separator
        
        # 如果没有检测结果，显示提示信息 (居中于内容区域)
        if self.hidden_result is None:
            msg = "等待隐藏情绪检测..."
            msg_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            msg_x = content_x + (content_width - msg_size[0]) // 2
            msg_y = content_y + content_height // 2 # Approx center Y
            self.put_text(canvas, msg, (msg_x, msg_y), (200, 200, 200), 16)
            return
        
        # --- 绘制隐藏情绪数据 (坐标基于 content_x 和 y_offset) ---
        is_hidden = self.hidden_result.is_hidden
        confidence = self.hidden_result.confidence
        macro_emotion = self.hidden_result.macro_emotion.value if self.hidden_result.macro_emotion else "未知"
        micro_emotion = self.hidden_result.micro_emotion.value if self.hidden_result.micro_emotion else "未知"
        hidden_emotion_type = self.hidden_result.hidden_emotion_type.value if self.hidden_result.hidden_emotion_type else "未知"
        supporting_aus = self.hidden_result.supporting_aus if hasattr(self.hidden_result, 'supporting_aus') else []
        
        # 根据是否检测到隐藏情绪设置标题颜色
        title_color = self.negative_color if is_hidden else self.positive_color
        title_text = "检测到隐藏情绪" if is_hidden else "未检测到隐藏情绪"
        self.put_text(canvas, title_text, 
                     (content_x + 20, y_offset), 
                     title_color, size=18)
        
        y_offset += 40
        
        # 显示置信度
        conf_text = f"置信度: {confidence:.2f}"
        conf_color = self.highlight_color if confidence > 0.6 else self.neutral_color
        self.put_text(canvas, conf_text, 
                     (content_x + 20, y_offset), 
                     conf_color, 16)
        
        y_offset += 30
        
        # 显示宏观和微观表情对比
        self.put_text(canvas, "情绪表现对比:", 
                     (content_x + 20, y_offset), 
                     self.text_color, 16)
        
        y_offset += 30
        
        # 绘制情绪对比区域背景 (坐标基于 content_x, y_offset)
        rect_x = content_x + 20
        rect_w = content_width - 40
        rect_h = 70
        cv2.rectangle(canvas, 
                     (rect_x, y_offset), 
                     (rect_x + rect_w, y_offset + rect_h), 
                     (40, 40, 45), -1)
        cv2.rectangle(canvas, 
                     (rect_x, y_offset), 
                     (rect_x + rect_w, y_offset + rect_h), 
                     (80, 80, 90), 1)
        
        # 绘制表格形式的情绪对比 (坐标基于 content_x, y_offset)
        table_y_start = y_offset
        # 表头
        self.put_text(canvas, "类型", 
                     (rect_x + 20, table_y_start + 20), 
                     (200, 200, 200), 14)
        self.put_text(canvas, "检测结果", 
                     (rect_x + 120, table_y_start + 20), # Adjust x offset for second column
                     (200, 200, 200), 14)
                     
        # 分隔线
        self.draw_horizontal_separator(canvas, 
                                      rect_x + 10, 
                                      table_y_start + 30, 
                                      rect_w - 20)
        
        # 宏观表情行
        self.put_text(canvas, "宏观情绪:", 
                     (rect_x + 20, table_y_start + 50), 
                     (170, 170, 170), 14)
        self.put_text(canvas, macro_emotion, 
                     (rect_x + 120, table_y_start + 50), 
                     (200, 200, 200), 14)
        
        # 微表情行 - 注意：这里原代码可能绘制在表格区域外，需调整
        # Table height is 70, row starts at y_offset+50. We need another row below.
        # Let's adjust the table layout slightly or move micro expression below.
        # For now, let's draw it just below the macro row, within the allocated 70 height.
        # It might overlap if text is long. A better layout might be needed.
        self.put_text(canvas, "微表情:", 
                     (rect_x + 20, table_y_start + 70), # Move down
                     (170, 170, 170), 14)
        self.put_text(canvas, micro_emotion, 
                     (rect_x + 120, table_y_start + 70), # Move down
                     (200, 200, 200), 14)
        
        y_offset += rect_h + 30 # Move past the comparison box + spacing
        
        # 如果检测到隐藏情绪，显示详细信息
        if is_hidden:
            # 显示隐藏的情绪类型
            self.put_text(canvas, f"隐藏情绪类型: {hidden_emotion_type}", 
                         (content_x + 20, y_offset), 
                         self.negative_color, 16)
            
            y_offset += 30
            
            # 绘制支持证据区域 (坐标基于 content_x, y_offset)
            evidence_rect_x = content_x + 20
            evidence_rect_w = content_width - 40
            evidence_rect_h = 150 # Allocate space for evidence
            # Check if there's enough space before drawing the box
            if y_offset + evidence_rect_h < content_y + content_height - 40: # Leave space for tip
                cv2.rectangle(canvas, 
                             (evidence_rect_x, y_offset), 
                             (evidence_rect_x + evidence_rect_w, y_offset + evidence_rect_h), 
                             (40, 40, 45), -1)
                cv2.rectangle(canvas, 
                             (evidence_rect_x, y_offset), 
                             (evidence_rect_x + evidence_rect_w, y_offset + evidence_rect_h), 
                             (80, 80, 90), 1)
                
                evidence_y_start = y_offset
                # 显示支持证据标题
                self.put_text(canvas, "支持证据 (AU):", 
                             (evidence_rect_x + 10, evidence_y_start + 20), 
                             (200, 200, 200), 15)
                
                evidence_y_offset = evidence_y_start + 40
                
                # 显示支持的AU列表
                if supporting_aus:
                    items_drawn = 0
                    for i, au_info in enumerate(supporting_aus):
                        if evidence_y_offset + 45 > evidence_y_start + evidence_rect_h: break # Stop if no space
                        items_drawn += 1
                        # ... (rest of AU info extraction) ...
                        au_id = str(au_info.id) if hasattr(au_info, 'id') else au_info.get("id", "")
                        au_name = au_info.name if hasattr(au_info, 'name') else au_info.get("name", "")
                        au_desc = au_info.description if hasattr(au_info, 'description') else au_info.get("description", "")
                        
                        au_text = f"AU{au_id}: {au_name}"
                        self.put_text(canvas, au_text, 
                                     (evidence_rect_x + 20, evidence_y_offset), 
                                     (180, 180, 220), 14)
                        
                        evidence_y_offset += 20
                        if evidence_y_offset + 25 > evidence_y_start + evidence_rect_h: break # Check again before drawing desc
                        # 显示AU描述（简短版本）
                        if len(au_desc) > 40:
                            au_desc = au_desc[:37] + "..."
                        self.put_text(canvas, f"  - {au_desc}", 
                                     (evidence_rect_x + 30, evidence_y_offset), 
                                     (150, 150, 180), 13)
                        
                        evidence_y_offset += 25
                        
                    if items_drawn == 0: # If loop didn't run or broke immediately
                         self.put_text(canvas, "无支持证据", 
                                  (evidence_rect_x + 20, evidence_y_offset), 
                                  (150, 150, 150), 14)
                else:
                    self.put_text(canvas, "无支持证据", 
                                 (evidence_rect_x + 20, evidence_y_offset), 
                                 (150, 150, 150), 14)
            # Update y_offset to be below the evidence box (even if not drawn fully)
            y_offset += evidence_rect_h + 10 

        # 绘制底部提示 (确保在面板内)
        tip_y = content_y + content_height - 20 # Position from bottom of content area
        info_text = "注: 基于微表情和宏观表情差异分析"
        # Ensure text fits horizontally
        self.put_text(canvas, info_text, 
                     (content_x + 20, tip_y), 
                     (150, 150, 150), 14)
        
        tip_y += 20
        method_text = "检测方法: 情绪一致性分析 + AU激活模式"
        self.put_text(canvas, method_text, 
                     (content_x + 20, tip_y), 
                     (150, 150, 150), 14) 