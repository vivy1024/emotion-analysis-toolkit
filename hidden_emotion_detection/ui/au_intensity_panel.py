#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - AU 综合面板 (状态与强度)
显示面部动作单元(AU)的激活状态和强度值，支持动态序列处理
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging # 添加 logging
import os

# 从 enhance_hidden.core 导入 EventBus 和 EventType
from ..core.event_bus import EventBus
from ..core.data_types import FrameResult, AUResult, Event, EventType
from .base_panel import BasePanel

logger = logging.getLogger(__name__) # 获取 logger

class AUIntensityPanel(BasePanel):
    """AU综合面板，显示AU状态和强度值"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化AU综合面板"""
        # 更新标题以反映合并后的内容
        super().__init__("面部动作单元(AU)状态/强度")
        
        # --- Colors ---
        self.text_color = (200, 200, 200)
        self.highlight_color = (100, 200, 250)
        self.au_border_color = (100, 100, 120)
        self.bar_bg_color = (60, 60, 65)
        self.au_active_color = (50, 200, 50)    # 激活的AU颜色（绿色）
        self.au_inactive_color = (50, 50, 60)   # 未激活颜色（深灰）
        
        # --- AU Definitions & Grouping (Keep as before) ---
        self.au_names = {
            "1": "内眉上扬",
            "2": "外眉上扬", 
            "4": "眉头下降",
            "5": "上眼睑抬起",
            "6": "脸颊抬起",
            "7": "眼睑紧绷",
            "9": "鼻子皱起",
            "10": "上唇抬起",
            "12": "嘴角拉伸",
            "14": "嘴角凹陷",
            "15": "嘴角下拉",
            "17": "下巴抬起",
            "20": "嘴角水平拉伸",
            "23": "嘴唇紧闭",
            "25": "唇部张开",
            "26": "下颌下降",
            "28": "唇部吸吮",
            "45": "眨眼"
        }
        self.au_groups = {
            "眉部": ["1", "2", "4"],
            "眼部": ["5", "7", "45"],
            "鼻部": ["9"],
            "上唇/脸颊": ["6", "10", "12", "14"],
            "下唇/下巴": ["15", "17", "20", "23", "25", "26", "28"],
        }
        
        # --- State Variables ---
        # Initialize both state and intensity dictionaries
        self.au_present = {au_id: False for au_id in self.au_names.keys()}
        self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
        self.au_intensities_raw = {au_id: 0.0 for au_id in self.au_names.keys()}
        self.last_update_time = 0
        
        # 动态序列信息
        self.is_sequence_result = False
        self.frames_count = 0
        self.sequence_update_time = 0
        self.sequence_type = 'unknown'
        self.au_model_name = 'default_model'

        # --- Event Handling ---
        self.event_bus = event_bus
        if self.event_bus:
            self.event_bus.subscribe(EventType.AU_ANALYZED, self._on_au_analyzed)
            logger.info("AUIntensityPanel已订阅AU_ANALYZED事件")
        else:
            logger.warning("No event bus provided to AUIntensityPanel, cannot subscribe to events.")
        # self.event_bus.subscribe(EventType.FACE_LOST, self._on_face_lost) # Optional
    
    def _on_au_analyzed(self, event: Event):
        """处理 AU_ANALYZED 事件，更新状态和强度"""
        try:
            # 使用 event.data 访问数据
            event_data = event.data
            if not isinstance(event_data, dict):
                logger.error(f"AUPanel received unexpected data type in event: {type(event_data)}")
                return
                
            logger.info(f"AUPanel received AU_ANALYZED event data keys: {list(event_data.keys())}")
            
            # 提取序列处理信息
            self.is_sequence_result = event_data.get('is_sequence', False)
            self.frames_count = event_data.get('frames_count', 0)
            self.sequence_type = event_data.get('sequence_type', 'unknown')
            self.au_model_name = event_data.get('au_model', 'default_model')
            
            if self.is_sequence_result:
                self.sequence_update_time = time.time()
                logger.info(f"AUPanel 接收到序列处理结果，类型：{self.sequence_type}，帧数：{self.frames_count}，模型：{self.au_model_name}")
            
            if event_data and 'result' in event_data:
                au_result = event_data.get('result')
                logger.info(f"AUPanel extracted au_result: {au_result is not None}")

                if au_result:
                    # --- 详细记录接收到的原始数据 ---
                    if hasattr(au_result, 'au_intensities_raw') and au_result.au_intensities_raw:
                        logger.info(f"原始强度值(au_intensities_raw): {au_result.au_intensities_raw}")
                        logger.info(f"原始强度值数量: {len(au_result.au_intensities_raw)}")
                    else:
                        logger.warning("接收到的AU结果中没有au_intensities_raw属性或为空")
                    
                    if hasattr(au_result, 'au_intensities') and au_result.au_intensities:
                        logger.info(f"标准化强度值(au_intensities): {au_result.au_intensities}")
                        logger.info(f"标准化强度值数量: {len(au_result.au_intensities)}")
                    else:
                        logger.warning("接收到的AU结果中没有au_intensities属性或为空")
                    
                    # --- 处理AU标识符的格式 ---
                    # feat返回的AU标识符可能是'AU01', 'AU02'等，需要转换为'1', '2'等
                    def normalize_au_key(key):
                        """将'AU01'等转换为'1'等"""
                        if isinstance(key, str) and key.startswith('AU'):
                            # 移除'AU'前缀，并移除前导零
                            return key[2:].lstrip('0')
                        return key
                    
                    # --- Update Normalized Intensities ---
                    if hasattr(au_result, 'au_intensities') and au_result.au_intensities:
                        # 转换AU键
                        normalized_intensities = {normalize_au_key(k): v for k, v in au_result.au_intensities.items()}
                        self.au_intensities = {k: v for k, v in normalized_intensities.items() if k in self.au_names}
                        logger.info(f"AUPanel updated intensities (normalized): { {k: f'{v:.2f}' for k,v in self.au_intensities.items()} }")
                        self.last_update_time = time.time()
                    else:
                        # Reset if no data
                        self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
                        logger.warning("AU结果中没有有效的au_intensities数据，重置为0")
                        
                    # --- Update Raw Intensities --- 
                    if hasattr(au_result, 'au_intensities_raw') and au_result.au_intensities_raw:
                        # 转换AU键
                        normalized_raw_intensities = {normalize_au_key(k): v for k, v in au_result.au_intensities_raw.items()}
                        self.au_intensities_raw = {k: v for k, v in normalized_raw_intensities.items() if k in self.au_names}
                        logger.info(f"AUPanel updated intensities (raw): { {k: f'{v:.2f}' for k,v in self.au_intensities_raw.items()} }")
                    else:
                        # 如果没有原始强度值但有标准化强度值，则尝试从标准化值计算
                        if hasattr(au_result, 'au_intensities') and au_result.au_intensities:
                            normalized_intensities = {normalize_au_key(k): v for k, v in au_result.au_intensities.items()}
                            self.au_intensities_raw = {k: v * 5.0 for k, v in normalized_intensities.items() if k in self.au_names}
                            logger.info(f"无原始强度值，从标准化值计算(×5): { {k: f'{v:.2f}' for k,v in self.au_intensities_raw.items()} }")
                        else:
                            # Reset if no data
                            self.au_intensities_raw = {au_id: 0.0 for au_id in self.au_names.keys()}
                            logger.warning("AU结果中没有有效的au_intensities_raw数据，重置为0")

                    # --- Update Presence --- 
                    if hasattr(au_result, 'au_present') and au_result.au_present:
                        # 转换AU键
                        normalized_presence = {normalize_au_key(k): v for k, v in au_result.au_present.items()}
                        self.au_present = {k: v for k, v in normalized_presence.items() if k in self.au_names}
                        logger.info(f"AUPanel updated presence: {self.au_present}")
                    else:
                        # Reset if no data
                        self.au_present = {au_id: False for au_id in self.au_names.keys()}
                        logger.warning("AU结果中没有有效的au_present数据，重置为False")
                else:
                    # Reset all if result object is None
                    self.au_present = {au_id: False for au_id in self.au_names.keys()}
                    self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
                    self.au_intensities_raw = {au_id: 0.0 for au_id in self.au_names.keys()}
                    logger.warning("AUPanel received event, but 'result' object was None. Resetting state.")
            else:
                # Reset all if no result key or empty data
                self.au_present = {au_id: False for au_id in self.au_names.keys()}
                self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
                self.au_intensities_raw = {au_id: 0.0 for au_id in self.au_names.keys()}
                logger.warning("AUPanel received event, but 'result' key was missing or event_data was empty. Resetting state.")
        except Exception as e:
            logger.error(f"AUPanel处理AU事件时出错: {e}", exc_info=True)
            # 重置所有状态
            self.au_present = {au_id: False for au_id in self.au_names.keys()}
            self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
            self.au_intensities_raw = {au_id: 0.0 for au_id in self.au_names.keys()}
    
    # Optional: Face lost handler
    # def _on_face_lost(self, event: Event):
    #     self.au_present = {au_id: False for au_id in self.au_names.keys()}
    #     self.au_intensities = {au_id: 0.0 for au_id in self.au_names.keys()}
    #     pass
    
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将合并后的AU状态和强度渲染到画布上
        """
        content_area = self.draw_panel_frame(canvas, x, y, width, height)
        content_x, content_y, content_width, content_height = content_area
        center_x = content_x + content_width // 2
        
        # --- Draw Header --- 
        description_height = 25 # Slightly smaller header
        # Remove the redundant title drawing here, BasePanel already draws it
        # title_x = center_x - text_width // 2
        # title_x = max(content_x + 5, title_x) 
        # title_y = content_y + 18 
        # self.put_text(canvas, self.title, 
        #              (title_x, title_y), 
        #              self.text_color, 13)
        
        # 显示序列模式状态
        if self.is_sequence_result and self.frames_count > 0:
            # 提取模型文件名，不显示路径
            model_short_name = os.path.basename(self.au_model_name) if '/' in self.au_model_name or '\\' in self.au_model_name else self.au_model_name
            
            # 格式化显示文本
            sequence_text = f"{self.sequence_type}序列分析 ({self.frames_count}帧) - 模型: {model_short_name}"
            text_size = cv2.getTextSize(sequence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # 计算位置 (右对齐)
            text_x = content_x + content_width - text_size[0] - 10
            text_y = content_y + 15
            
            # 优化显示效果：添加背景区域提高可读性
            cv2.rectangle(canvas, 
                          (text_x - 5, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 5, text_y + 3),
                          (50, 70, 50), -1)
            
            # 绘制文本
            cv2.putText(canvas, sequence_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1, cv2.LINE_AA)
                     
        # Draw separator line slightly lower if the internal title is removed
        cv2.line(canvas, 
                (content_x + 5, content_y + description_height -5), # Adjusted y position
                (content_x + content_width - 5, content_y + description_height - 5), # Adjusted y position
                (80, 80, 80), 1)
        # Adjust background rectangle start y
        cv2.rectangle(canvas, 
                     (content_x + 2, content_y + description_height), # Adjusted y position
                     (content_x + content_width - 2, content_y + content_height - 2), 
                     (40, 40, 45), -1) # Background
                
        # --- Layout Parameters for Combined Row --- 
        # Adjust initial offset based on removed internal title and separator adjustment
        y_offset = content_y + description_height + 4 # Start drawing below separator (was +12)
        group_spacing = 10 # Spacing between groups
        group_header_height = 18 # Group header height
        
        # === Increased Vertical Space ===
        row_height = 16 # Significantly increase row height (was 10)
        row_spacing = 6   # Increase vertical space between AU rows (was 3)
        
        # === X Coordinate Calculation (Review and Adjust Spacing) ===
        indicator_size = 12 
        indicator_offset_x = content_x + 8 # Slightly reduce start indent
        indicator_right_x = indicator_offset_x + indicator_size
        
        label_offset_x = indicator_right_x + 6 # Space after indicator (was 8)
        # Reduce label max width to give more space for bar/value
        label_max_width = 75  # Shorter label space (was 90)
        label_right_x = label_offset_x + label_max_width
        
        # === Increased Bar Width ===
        bar_offset_x = label_right_x + 6 # Space after label (was 8)
        value_text_width = 35 
        # Recalculate bar_max_width trying to make it longer
        # Reduce end margin slightly
        bar_max_width = content_width - (bar_offset_x - content_x) - value_text_width - 10 # (was 15 end margin)
        bar_max_width = max(20, bar_max_width) # Ensure a minimum bar width
        bar_right_x = bar_offset_x + bar_max_width
        
        value_text_offset_x = bar_right_x + 4 # Space before value text (was 5)
        
        # === Font and Text Alignment ===
        font_size = 11 
        group_font_size = 12 
        # Adjust text_offset_y based on the new row_height
        text_offset_y = row_height // 2 + 4 # May need fine-tuning
        
        drawn_au_count = 0
        
        # --- Draw AU Groups and Rows ---
        for group_name, au_ids in self.au_groups.items():
            if y_offset + group_header_height > content_y + content_height: break
            
            # Draw group header
            self.put_text(canvas, f"{group_name}:", 
                         (content_x + 10, y_offset + group_header_height - 5), 
                         self.highlight_color, group_font_size)
            y_offset += group_header_height
            # === Add extra space after group header ===
            y_offset += 5 # Add 5 pixels gap
            # ==========================================
            
            # Draw AU rows for this group
            for au_id in au_ids:
                # Check vertical space *before* drawing the row
                if y_offset + row_height + row_spacing > content_y + content_height: break
                
                is_active = self.au_present.get(au_id, False)
                
                # 1. Draw State Indicator (Circle)
                indicator_center_y = y_offset + row_height // 2
                indicator_color = self.au_active_color if is_active else self.au_inactive_color
                # Use indicator_offset_x and indicator_center_y
                cv2.circle(canvas, (indicator_offset_x + indicator_size // 2, indicator_center_y), 
                          indicator_size // 2, indicator_color, -1)
                cv2.circle(canvas, (indicator_offset_x + indicator_size // 2, indicator_center_y), 
                          indicator_size // 2, self.au_border_color, 1)
                if is_active:
                    cv2.circle(canvas, (indicator_offset_x + indicator_size // 2, indicator_center_y), 
                              1, (255, 255, 255), -1)

                # 2. Draw AU Name (Truncated if needed)
                au_name_text = f"AU{au_id}: {self.au_names.get(au_id, '')}"
                # Truncate text logic
                estimated_char_width = font_size * 0.6 # Approximate char width
                # Use label_max_width for truncation calculation
                max_chars = int(label_max_width / estimated_char_width) if estimated_char_width > 0 else 0
                if max_chars > 3 and len(au_name_text) > max_chars:
                     au_name_text = au_name_text[:max_chars-3] + "..."
                # Use label_offset_x and calculated y offset 
                self.put_text(canvas, au_name_text, 
                            (label_offset_x, y_offset + text_offset_y), 
                            self.text_color if is_active else (150, 150, 150), font_size)
                
                # 3. Draw Intensity Bar (using RAW intensity for bar length and color)
                raw_intensity_for_bar = self.au_intensities_raw.get(au_id, 0.0)
                max_expected_intensity = 5.0 # Define the max expected raw intensity for normalization

                # Normalize raw intensity to 0.0-1.0 for bar width and color calculation
                # Ensure raw_intensity_for_bar is not negative before division
                normalized_intensity_for_bar = max(0.0, min(1.0, raw_intensity_for_bar / max_expected_intensity if raw_intensity_for_bar > 0 else 0.0))
                
                bar_width = int(normalized_intensity_for_bar * bar_max_width)
                bar_width = min(bar_width, bar_max_width) # Ensure it doesn't exceed max width
                
                # Background - Use bar_offset_x
                cv2.rectangle(canvas, 
                             (bar_offset_x, y_offset), 
                             (bar_offset_x + bar_max_width, y_offset + row_height), 
                             self.bar_bg_color, -1)
                # Border - Use bar_offset_x
                cv2.rectangle(canvas, 
                             (bar_offset_x, y_offset), 
                             (bar_offset_x + bar_max_width, y_offset + row_height), 
                             self.au_border_color, 1)
                # Foreground (colored bar) - Use bar_offset_x
                if bar_width > 0:
                    # Use normalized_intensity_for_bar for color calculation as well
                    r = int(min(255, 510 * normalized_intensity_for_bar))       
                    g = int(min(255, 510 * (1 - normalized_intensity_for_bar))) 
                    b = 0
                    color = (b, g, r)
                    cv2.rectangle(canvas, 
                                 (bar_offset_x, y_offset), 
                                 (bar_offset_x + bar_width, y_offset + row_height), 
                                 color, -1)
                    
                # 4. Draw Intensity Value Text (Show Raw Value)
                raw_intensity_val = self.au_intensities_raw.get(au_id, 0.0) # Use raw intensity
                
                # 确保显示原始值(0-5范围)，而不是标准化值(0-1范围)
                # 如果值非常小但非零，可能是标准化值，需要转换
                if 0 < raw_intensity_val < 0.2 and self.au_intensities.get(au_id, 0.0) > 0:
                    # 可能是标准化值，转换为原始值
                    raw_intensity_val = self.au_intensities.get(au_id, 0.0) * 5.0
                    logger.info(f"AU{au_id}强度值过小({raw_intensity_val:.2f})可能是标准化值，转换为: {raw_intensity_val:.2f}")
                
                # 确保值在0-5范围内
                raw_intensity_val = max(0.0, min(5.0, raw_intensity_val))
                
                # 在绘制时突出显示活跃的AU
                if is_active or raw_intensity_val > 0.5:
                    intensity_text = f"{raw_intensity_val:.2f}" # Display raw value with 2 decimals
                    text_color = self.text_color
                    # 根据强度值调整颜色
                    if raw_intensity_val > 3.0:
                        text_color = (50, 255, 50)  # 高强度显示亮绿色
                    elif raw_intensity_val > 1.5:
                        text_color = (150, 255, 150)  # 中等强度显示浅绿色
                else:
                    intensity_text = f"{raw_intensity_val:.2f}" # Display raw value with 2 decimals
                    text_color = (150, 150, 150)  # 低强度/非活跃使用灰色
                
                self.put_text(canvas, intensity_text, 
                           (value_text_offset_x, y_offset + text_offset_y), 
                           text_color, font_size)
                
                # Move to next row
                y_offset += row_height + row_spacing
                drawn_au_count += 1
            
            if y_offset + row_height + row_spacing > content_y + content_height: break
            if y_offset + group_spacing > content_y + content_height: break
            y_offset += group_spacing
            
            # Draw group separator line
            if group_name != list(self.au_groups.keys())[-1] and y_offset < content_y + content_height:
                cv2.line(canvas, 
                        (content_x + 5, y_offset - group_spacing // 2), 
                        (content_x + content_width - 5, y_offset - group_spacing // 2), 
                        (60, 60, 60), 1)
        
        # --- Draw Overflow/Filtering Indicator --- 
        if drawn_au_count < len(self.au_names):
            # Some AUs were not drawn due to space limit
            remaining_aus = len(self.au_names) - drawn_au_count
            tip_y = content_y + content_height - 15
            if tip_y > y_offset:
                tip_text = f"(...{remaining_aus} more AUs)" # Simplified text
                self.put_text(canvas, tip_text, (content_x + content_width - 150, tip_y), (120, 120, 120), 12)
        # Remove the elif block that referenced intensity_threshold
        # elif len(self.au_intensities) > active_au_count:
        #      # Indicate that some AUs were filtered by intensity
        #      filtered_count = len(self.au_names) - active_au_count
        #      tip_y = content_y + content_height - 15
        #      if tip_y > y_offset:
        #           tip_text = f"({filtered_count} AUs < {intensity_threshold:.2f})"
        #           self.put_text(canvas, tip_text, (content_x + content_width - 150, tip_y), (100, 100, 100), 11) 