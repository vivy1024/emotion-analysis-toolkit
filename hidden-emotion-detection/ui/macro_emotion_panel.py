#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 宏观情绪面板
显示宏观情绪分析结果，包含AU辅助情绪区域
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time # 确保导入 time
import os # 确保导入 os

# --- 创建专门记录分析结果的 Logger ---
results_log_file = 'logs/emotion_analysis_results.log'
os.makedirs(os.path.dirname(results_log_file), exist_ok=True) # 确保logs目录存在

results_logger = logging.getLogger('EmotionResults')
results_logger.setLevel(logging.INFO) # 只记录 INFO 及以上级别

# 防止重复添加 handler (如果模块被多次加载)
if not results_logger.handlers:
    # 创建文件处理器 (使用 'w' 模式覆盖文件)
    file_handler = logging.FileHandler(results_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到 Logger
    results_logger.addHandler(file_handler)
    results_logger.propagate = False # 防止结果日志也输出到控制台（如果根logger有ConsoleHandler）

# --- 结束 Logger 设置 ---

from .base_panel import BasePanel
from ..core.data_types import EmotionType, EmotionResult, AUResult
from ..core.event_bus import EventBus, Event, EventType

logger = logging.getLogger(__name__.split('.')[-1].upper())

class MacroEmotionPanel(BasePanel):
    """宏观情绪面板，显示宏观情绪分析结果"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化宏观情绪面板"""
        super().__init__("宏观表情分析")
        self.event_bus = event_bus
        logger.info(f"MacroEmotionPanel initialized. Event bus provided: {self.event_bus is not None}")
        
        # 情绪颜色映射
        self.emotion_colors = {
            EmotionType.HAPPINESS: (0, 255, 0),    # 高兴：绿色
            EmotionType.DISGUST: (0, 140, 255),    # 厌恶：橙色
            EmotionType.SURPRISE: (255, 255, 0),   # 惊讶：青色
            EmotionType.FEAR: (255, 0, 255),       # 恐惧：紫色
            EmotionType.SADNESS: (255, 128, 0),    # 悲伤：蓝色
            EmotionType.ANGER: (0, 0, 255),        # 愤怒：红色
            EmotionType.NEUTRAL: (128, 128, 128),  # 中性：灰色
            EmotionType.CONTEMPT: (160, 0, 160),   # 蔑视：深紫色
            EmotionType.REPRESSION: (0, 180, 180), # 压抑：青色
            EmotionType.CONFUSION: (0, 180, 180),  # 困惑：同压抑一样的青色
            EmotionType.UNKNOWN: (200, 200, 200)   # 未知：浅灰色
        }
        
        # 情绪分析结果
        self.emotion_result: Optional[EmotionResult] = None
        self.last_update_time = 0 # 添加 last_update_time
        
        # 添加最终分析结果
        self.final_emotion_result: Optional[EmotionResult] = None
        self.final_last_update_time = 0
        
        # 添加主引擎结果
        self.raw_emotion_result: Optional[EmotionResult] = None
        self.raw_last_update_time = 0
        
        # 添加AU辅助情绪结果
        self.au_suggestions = {}  # 初始化为空字典而不是None
        self.last_au_update_time: float = 0
        self.last_au_source: str = ""
        
        # 添加数据稳定性相关变量
        self.display_data = {}  # 当前显示的数据
        self.forced_display_interval = 1.2  # 强制显示间隔，比微表情更长
        self.min_display_time = 2.5  # 数据最小显示时间，比微表情更长
        self.last_valid_data = None  # 上次有效数据
        self.last_valid_data_time = 0  # 上次有效数据时间
        
        # --- 订阅事件 ---
        if self.event_bus:
            try:
                # 订阅三个主要事件
                self.event_bus.subscribe(EventType.RAW_MACRO_EMOTION_ANALYZED, self._on_macro_emotion_analyzed) # 主引擎结果
                self.event_bus.subscribe(EventType.AU_MACRO_EMOTION_ANALYZED, self._on_macro_emotion_analyzed)  # AU辅助引擎结果
                self.event_bus.subscribe(EventType.MACRO_EMOTION_ANALYZED, self._on_macro_emotion_analyzed)     # 最终整合结果
                # 订阅AU辅助情绪事件
                self.event_bus.subscribe(EventType.AU_ANALYZED, self._on_au_analyzed)
                logger.info("MacroEmotionPanel subscribed to RAW_MACRO_EMOTION_ANALYZED, AU_MACRO_EMOTION_ANALYZED, MACRO_EMOTION_ANALYZED and AU_ANALYZED events.")
            except Exception as e:
                 logger.error(f"Failed to subscribe MacroEmotionPanel to event: {e}", exc_info=True)
        else:
            logger.warning("No event bus provided to MacroEmotionPanel, cannot subscribe to events.")
    
    def _on_au_analyzed(self, event: Event):
        """处理AU分析事件，获取AU辅助情绪建议"""
        try:
            # 获取事件数据
            result_data = event.data
            if not result_data:
                logger.warning("MacroEmotionPanel: 收到空的AU分析事件数据")
                return
            
            logger.info(f"MacroEmotionPanel: 收到AU_ANALYZED事件, 数据键: {list(result_data.keys()) if isinstance(result_data, dict) else '非字典类型'}")
                
            # 是否为UI更新专用事件
            is_ui_update = result_data.get('update_ui', False)
            
            # 1. 直接从AU引擎结果获取数据
            if 'result' in result_data and isinstance(result_data['result'], AUResult):
                au_result = result_data['result']
                logger.info(f"MacroEmotionPanel: 从AU结果获取数据，AU强度: {len(au_result.au_intensities)} 项")
                
                # 转换AU强度为情绪映射
                try:
                    # 创建模拟情绪数据
                    emotions_data = {
                        "happiness": max(0.0, min(1.0, au_result.au_intensities.get("12", 0.0) * 2.0)),  # AU12 - 嘴角上扬
                        "sadness": max(0.0, min(1.0, au_result.au_intensities.get("15", 0.0) * 1.5)),   # AU15 - 嘴角下垂
                        "surprise": max(0.0, min(1.0, (au_result.au_intensities.get("1", 0.0) + 
                                                     au_result.au_intensities.get("2", 0.0) + 
                                                     au_result.au_intensities.get("5", 0.0)) / 3 * 1.5)),  # AU1+2+5 - 眉毛上扬+眼睛睁大
                        "fear": max(0.0, min(1.0, (au_result.au_intensities.get("1", 0.0) + 
                                                 au_result.au_intensities.get("4", 0.0)) / 2 * 1.5)),  # AU1+4 - 内眉上扬+眉头皱起
                        "anger": max(0.0, min(1.0, au_result.au_intensities.get("4", 0.0) * 1.8)),  # AU4 - 眉头皱起
                        "disgust": max(0.0, min(1.0, au_result.au_intensities.get("9", 0.0) * 2.0)),  # AU9 - 鼻子皱起
                        "contempt": max(0.0, min(1.0, au_result.au_intensities.get("14", 0.0) * 1.5)),  # AU14 - 嘴角凹陷
                        "neutral": 0.5  # 默认中性值
                    }
                    
                    # 调整中性值 - 如果有强情绪则降低中性值
                    max_emotion = max([v for k, v in emotions_data.items() if k != "neutral"])
                    emotions_data["neutral"] = max(0.0, min(0.8, 0.8 - max_emotion * 0.8))
                    
                    # 添加序列标记
                    emotions_data['is_sequence'] = result_data.get('is_sequence', False)
                    emotions_data['sequence_length'] = result_data.get('sequence_length', 0)
                    
                    logger.info(f"MacroEmotionPanel: 从AU强度生成情绪映射: {', '.join([f'{k}={v:.2f}' for k, v in emotions_data.items() if k not in ['is_sequence', 'sequence_length']])}")
                    
                    # 立即更新AU建议
                    self.au_suggestions = emotions_data.copy()
                    self.last_au_update_time = time.time()
                    self.last_au_source = "AU强度映射"
                    return
                except Exception as e:
                    logger.error(f"从AU强度生成情绪映射失败: {e}", exc_info=True)
            
            # 2. 尝试从AU情绪引擎组件获取数据
            au_emotion_engine = result_data.get('components', {}).get('au_emotion_engine')
            
            # 如果直接从组件字典中无法获取 au_emotion_engine，尝试从 MacroEmotionAUEngine 获取数据
            if not au_emotion_engine and 'components' in result_data:
                macro_au_engine = result_data.get('components', {}).get('macro_emotion_au_engine')
                if macro_au_engine:
                    logger.info(f"MacroEmotionPanel: 直接从 macro_emotion_au_engine 获取数据")
                    # 直接保存 AU 辅助情绪结果，无需额外处理
                    self.au_suggestions = macro_au_engine.get_all_macro_emotions()
                    self.last_au_update_time = time.time()
                    self.last_au_source = "宏观AU引擎"
                    return
            
            # 如果是UI更新专用事件，跳过时间间隔检查
            if not is_ui_update:
                # 检查强制间隔 - 如果距离上次更新时间太短，跳过以避免频繁更新
                current_time = time.time()
                if current_time - self.last_au_update_time < self.forced_display_interval:
                    return
            
            # 获取宏观情绪辅助引擎的所有情绪建议
            if au_emotion_engine:
                emotions_data = au_emotion_engine.get_all_emotional_suggestions(is_micro=False)
                if emotions_data:
                    logger.info(f"MacroEmotionPanel: 获取到AU辅助情绪数据: {list(emotions_data.keys())}")
                    self.last_au_source = "情绪辅助引擎"
                else:
                    logger.debug("MacroEmotionPanel: AU引擎情绪建议为空")
                    return
            else:
                # 尝试从 result_data 中直接获取 all_probabilities
                if isinstance(result_data.get('result'), EmotionResult) and result_data.get('source') == 'AUEmotionEngine':
                    raw_probs = result_data.get('raw_probabilities')
                    if raw_probs:
                        logger.info(f"MacroEmotionPanel: 从 raw_probabilities 获取数据: {list(raw_probs.keys())}")
                        emotions_data = raw_probs.copy()
                        # 添加序列标记
                        emotions_data['is_sequence'] = result_data.get('is_sequence', False)
                        emotions_data['sequence_length'] = result_data.get('sequence_length', 0)
                        self.last_au_source = "AU情绪引擎"
                    else:
                        logger.debug("MacroEmotionPanel: raw_probabilities 不可用")
                        return
                else:
                    logger.debug("MacroEmotionPanel: 无法获取 AU 辅助情绪数据")
                    return
            
            # 如果没有有效数据，保持使用上一次有效数据
            has_valid_data = False
            for emotion_name, confidence in emotions_data.items():
                if emotion_name not in ['neutral', 'is_sequence', 'sequence_length'] and confidence > 0.1:
                    has_valid_data = True
                    break
                    
            if has_valid_data:
                # 有有效数据，更新显示和缓存
                self.display_data = emotions_data.copy()
                self.last_valid_data = emotions_data.copy()
                self.last_valid_data_time = time.time()
                # 更新 au_suggestions 以便在渲染时使用
                self.au_suggestions = emotions_data.copy()
                logger.info(f"MacroEmotionPanel: 更新 AU 建议数据，包含 {len(emotions_data)} 个项目")
            elif self.last_valid_data:
                # 没有有效数据但有缓存，检查是否超出最小显示时间
                if time.time() - self.last_valid_data_time > self.min_display_time:
                    # 超出最小时间，使用更平滑的衰减
                    decay_factor = max(0.6, 1.0 - (time.time() - self.last_valid_data_time) / 25.0)
                    
                    # 复制原数据并应用衰减
                    self.display_data = {k: v * decay_factor if k not in ['neutral', 'is_sequence', 'sequence_length'] 
                                       else v for k, v in self.last_valid_data.items()}
                    
                    # 增加neutral值补偿总和
                    if 'neutral' in self.display_data:
                        self.display_data['neutral'] = min(0.9, self.display_data.get('neutral', 0) + 0.04)
                        
                    # 更新 au_suggestions
                    self.au_suggestions = self.display_data.copy()
                else:
                    # 在最小显示时间内，保持数据完全不变
                    self.display_data = self.last_valid_data.copy()
                    self.au_suggestions = self.last_valid_data.copy()
            
            # 更新时间戳
            self.last_au_update_time = time.time()
            
        except Exception as e:
            logger.error(f"处理AU分析事件出错: {e}", exc_info=True)
    
    def _on_macro_emotion_analyzed(self, event: Event):
        """处理接收到的宏观情绪分析事件，区分主引擎和AU辅助引擎结果"""
        try:
            valid_event_types = [
                EventType.RAW_MACRO_EMOTION_ANALYZED,  # 主引擎原始结果
                EventType.AU_MACRO_EMOTION_ANALYZED,   # AU辅助引擎结果
                EventType.MACRO_EMOTION_ANALYZED       # 最终整合结果
            ]
            
            if event.type not in valid_event_types or 'result' not in event.data:
                logger.debug(f"收到事件，但不是有效的宏观情绪事件类型，或缺少'result'键。类型: {event.type}")
                return
                
            result_data = event.data['result']
            source = event.data.get('source', 'main_engine')  # 默认源
            
            # 检查是否是微表情结果，如果是，则忽略
            if getattr(result_data, 'is_micro_expression', False):
                logger.debug(f"忽略微表情结果: {result_data.emotion_type.name}")
                return
                
            if not isinstance(result_data, EmotionResult):
                logger.warning(f"收到{event.type}事件，但结果数据不是EmotionResult类型: {type(result_data)}")
                return
                
            # 记录处理信息 - 使用INFO级别确保可见
            logger.info(f"处理宏观情绪: {result_data.emotion_type.name}, 概率: {result_data.probability:.2f}, 来源: {source}, 事件类型: {event.type}")
            
            # 根据事件类型和来源处理结果
            if event.type == EventType.RAW_MACRO_EMOTION_ANALYZED:
                # 处理来自主引擎的原始结果
                probability_threshold = 0.60
                
                # 保存主引擎结果供显示
                self.raw_emotion_result = result_data
                self.raw_last_update_time = time.time()
                
                if result_data.probability >= probability_threshold:
                    # 接受主引擎结果
                    self.emotion_result = result_data
                    # 移除可能存在的AU引擎标记
                    if hasattr(self.emotion_result, 'from_au_engine'):
                        delattr(self.emotion_result, 'from_au_engine')
                        
                    # 记录到专用日志
                    results_logger.info(f"MACRO_RAW - 情绪: {str(self.emotion_result.emotion_type)}, 概率: {self.emotion_result.probability:.4f}")
                    self.last_update_time = time.time()
                    logger.info(f"宏观情绪面板更新为主引擎结果: {str(self.emotion_result.emotion_type)} ({self.emotion_result.probability:.2f})")
                else:
                    # 清除长时间未更新的结果
                    if self.emotion_result is not None and (time.time() - self.last_update_time) > 5.0:
                        logger.info(f"清除过期的宏观情绪结果")
                        self.emotion_result = None
                    logger.info(f"主引擎宏观情绪低于阈值 ({result_data.probability:.2f} < {probability_threshold})，可能使用AU辅助结果")
            
            elif event.type == EventType.AU_MACRO_EMOTION_ANALYZED:
                # 处理来自AU辅助引擎的结果
                logger.info(f"收到AU辅助宏观情绪分析结果: {result_data.emotion_type} ({result_data.probability:.2f})")
                
                # 放宽后备结果条件
                use_fallback = (
                    self.emotion_result is None or               # 无现有结果
                    (time.time() - self.last_update_time) > 1.0 or  # 主引擎超时
                    result_data.probability >= 0.7               # AU结果置信度高
                )
                
                if use_fallback:
                    self.emotion_result = result_data
                    # 标记来源
                    setattr(self.emotion_result, 'from_au_engine', True)
                    # 记录使用AU辅助结果
                    results_logger.info(f"AU辅助宏观 - 情绪: {str(self.emotion_result.emotion_type)}, 概率: {self.emotion_result.probability:.4f}")
                    logger.info(f"宏观情绪面板使用AU辅助结果: {str(self.emotion_result.emotion_type)} ({self.emotion_result.probability:.2f})")
                else:
                    logger.info(f"忽略AU辅助建议 - 存在最近的主引擎结果 (上次更新: {time.time() - self.last_update_time:.2f}秒前)")
            
            elif event.type == EventType.MACRO_EMOTION_ANALYZED:
                # 处理最终整合结果
                logger.info(f"收到最终宏观情绪分析结果: {result_data.emotion_type} ({result_data.probability:.2f})")
                # 保存最终分析结果
                self.final_emotion_result = result_data
                self.final_last_update_time = time.time()
                
                # 同时更新通用结果
                self.emotion_result = result_data
                self.last_update_time = time.time()
                
                results_logger.info(f"MACRO_FINAL - 情绪: {str(self.emotion_result.emotion_type)}, 概率: {self.emotion_result.probability:.4f}")
        
        except Exception as e:
            logger.error(f"处理宏观情绪事件出错: {e}", exc_info=True)
    
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上。先使用 BasePanel 的 render 绘制框架，再绘制动态情绪内容。
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
            # logger is not defined in this scope, use logging directly or add self.logger if BasePanel has it
            logging.error(f"Error during BasePanel render call from MacroEmotionPanel: {e}")
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

        # --- 先添加最终分析结果区域 ---
        final_result_height = 40  # 最终结果区域高度
        
        # 绘制分隔线和标题
        cv2.rectangle(canvas, 
                    (content_x, content_y), 
                    (content_x + content_width, content_y + final_result_height), 
                    (45, 45, 60), -1)

        # 显示最终分析结果
        if self.final_emotion_result:
            final_emotion_text = f"最终情绪: {str(self.final_emotion_result.emotion_type)}"
            final_prob_text = f"置信度: {self.final_emotion_result.probability:.2f}"
            final_source = self.final_emotion_result.source if hasattr(self.final_emotion_result, 'source') else "整合引擎"
            final_source_text = f"来源: {final_source}"
            
            # 在最终结果区域显示情绪和置信度
            self.put_text(canvas, final_emotion_text, (content_x + 10, content_y + 20), (0, 255, 255), 16)
            self.put_text(canvas, final_prob_text, (content_x + 200, content_y + 20), (0, 255, 255), 16)
            self.put_text(canvas, final_source_text, (content_x + 300, content_y + 20), (150, 150, 220), 14)
        else:
            # 显示空背景，不显示任何文字
            cv2.rectangle(canvas, (content_x, content_y), (content_x + content_width, content_y + final_result_height), (40, 40, 45), -1)
            
        # --- 调整剩余区域的位置 ---
        # 原内容区域的起始位置下移
        content_y += final_result_height + 5
        content_height -= final_result_height + 5
        
        # --- 渲染主引擎结果区域 ---
        cv2.rectangle(canvas, 
                    (content_x, content_y), 
                    (content_x + content_width, content_y + 30), 
                    (45, 45, 60), -1)
        
        # 显示主引擎来源 - 使用14号字体保持一致性
        model_name = "主要检测模型"
        source_text = f"来源: 宏表情主引擎"
        self.put_text(canvas, source_text, (content_x + 10, content_y + 20), (150, 150, 220), 14)
            
        content_y += 30  # 下移到主引擎区域下方

        # --- 定义固定的情绪显示顺序 ---
        ordered_emotion_types = [
            EmotionType.ANGER,      # 愤怒
            EmotionType.DISGUST,    # 厌恶
            EmotionType.FEAR,       # 恐惧
            EmotionType.HAPPINESS,  # 高兴
            EmotionType.SADNESS,    # 悲伤
            EmotionType.SURPRISE,   # 惊讶
            EmotionType.NEUTRAL     # 中性
        ]

        # 如果没有检测结果，显示空白区域
        if self.emotion_result is None:
            # 显示空白区域，不显示任何文字
            cv2.rectangle(canvas, (content_x, content_y), (content_x + content_width, content_y + 30), (35, 35, 40), -1)
            dominant_emotion = None
            dominant_prob = 0
            # 创建默认情绪数据，只显示面板结构
            emotion_probs = {emotion_type: 0.0 for emotion_type in ordered_emotion_types}
        else:
            # 情绪概率字典，初始化为0
            emotion_probs = {emotion_type: 0.0 for emotion_type in ordered_emotion_types}

            # 提取情绪和概率
            dominant_emotion = self.emotion_result.emotion_type
            dominant_prob = self.emotion_result.probability

            # 更新情绪概率 - 注意: emotion_result 可能只包含主导情绪
            if hasattr(self.emotion_result, 'all_probabilities') and self.emotion_result.all_probabilities:
                 for em_type, prob in self.emotion_result.all_probabilities.items():
                     if em_type in emotion_probs:
                          emotion_probs[em_type] = prob
            else: # Fallback: only show dominant probability
                 if dominant_emotion in emotion_probs:
                    emotion_probs[dominant_emotion] = dominant_prob

        # 设置动态内容的 Y 轴起始位置
        y_offset = content_y + 20 # Start near top of content area

        # 显示主导情绪文本
        # 检查是否来自AU引擎
        source_indicator = ""
        if self.emotion_result and hasattr(self.emotion_result, 'from_au_engine') and getattr(self.emotion_result, 'from_au_engine', False):
            source_indicator = " (AU辅助)"
        
        # 只在有情绪结果时显示主导情绪
        if self.emotion_result:
            dominant_text = f"当前情绪: {str(dominant_emotion)}{source_indicator} ({dominant_prob:.2f})"
            # Position relative to content area
            self.put_text(canvas, dominant_text, (content_x + 10, y_offset), (0, 255, 255), 16)

        y_offset += 5 # <-- 极限减小主导情绪下方的间距

        # 绘制所有情绪的概率柱状图
        bar_start_y = y_offset
        bar_height = 15 # <-- 减小高度
        bar_spacing = 4 # <-- 减小间距
        bar_label_width = 55 # <-- 显著减小标签宽度
        bar_value_width = 40 # Space for probability value
        # --- 重新计算 bar_max_width, 减少标签和条之间的间距 --- 
        space_between_label_bar = 5 
        space_between_bar_value = 5
        bar_max_width = content_width - bar_label_width - bar_value_width - space_between_label_bar - space_between_bar_value - 10 # -10 for initial content_x padding
        bar_max_width = max(10, bar_max_width) # Ensure positive width

        # Check if enough vertical space for bars
        required_bar_height = len(ordered_emotion_types) * (bar_height + bar_spacing)
        if bar_start_y + required_bar_height > content_y + content_height / 2 - 20: # 给AU辅助部分预留空间
            self.put_text(canvas, "空间不足", (content_x + 10, bar_start_y), (180, 180, 0), 14)
        else:
            for i, emotion_type in enumerate(ordered_emotion_types):
                # 情绪名称和概率文本
                emotion_text = str(emotion_type)
                prob = emotion_probs.get(emotion_type, 0.0)
                prob_text = f"{prob:.2f}"

                # 计算柱状图位置 (absolute coordinates)
                bar_y = bar_start_y + i * (bar_height + bar_spacing)
                label_x = content_x + 10
                bar_x = label_x + bar_label_width + space_between_label_bar # <-- 使用新间距
                value_x = bar_x + bar_max_width + space_between_bar_value # <-- 使用新间距

                # 绘制情绪名称 (中文)
                self.put_text(canvas, emotion_text, (label_x, bar_y + bar_height // 2 + 4), (200, 200, 200), 13)

                # 计算概率条长度
                bar_width_actual = int(prob * bar_max_width)
                bar_width_actual = max(0, min(bar_width_actual, bar_max_width))

                # 设置颜色
                bar_color = self.emotion_colors.get(emotion_type, (128, 128, 128))
                # Highlight dominant emotion bar maybe?
                if emotion_type == dominant_emotion:
                     border_color = (255, 255, 255) # White border for dominant
                else:
                     border_color = (70, 70, 75) # Standard separator color

                # 绘制概率背景
                cv2.rectangle(canvas,
                             (bar_x, bar_y),
                             (bar_x + bar_max_width, bar_y + bar_height),
                             (60, 60, 65), -1)

                # 绘制概率条
                if bar_width_actual > 0:
                    cv2.rectangle(canvas,
                                 (bar_x, bar_y),
                                 (bar_x + bar_width_actual, bar_y + bar_height),
                                 bar_color, -1)
                
                # Draw border around the bar area
                cv2.rectangle(canvas, 
                              (bar_x, bar_y), 
                              (bar_x + bar_max_width, bar_y + bar_height), 
                              border_color, 1)


                # 绘制概率文本
                self.put_text(canvas, prob_text,
                             (value_x, bar_y + bar_height // 2 + 4),
                             (200, 200, 200), 12)
        
        # --- 添加AU辅助情绪部分 ---
        # 计算AU辅助部分的位置
        au_section_y = content_y + content_height // 2
        
        # 绘制分隔线和背景框
        cv2.rectangle(canvas, 
                    (content_x, au_section_y), 
                    (content_x + content_width, content_y + content_height - 5), 
                    (40, 40, 50), -1)
        
        # 添加AU辅助数据来源指示 - 使用与上方相同的字体大小
        au_source = "来源: AU强度映射"
        self.put_text(canvas, au_source, (content_x + 10, au_section_y + 20), (150, 150, 220), 14)
        
        # 如果没有AU辅助情绪数据，显示空白区域，不再显示等待文字
        if not self.au_suggestions or time.time() - self.last_au_update_time > 3.0:
            # 创建空的情绪字典，保证后续代码有数据可用
            emotion_dict = {}
        else:
            # 过滤掉非情绪类型的键
            emotion_dict = {k: v for k, v in self.au_suggestions.items() 
                          if k not in ['is_sequence', 'sequence_length']}
        
        # 定义完整FACS系统的宏观情绪辅助情绪显示顺序
        # FACS系统情绪：Happiness, Sadness, Surprise, Fear, Anger, Disgust, Contempt
        au_display_emotion_types = [
            "happiness", "sadness", "surprise", "fear", "anger", 
            "disgust", "contempt", "neutral"
        ]
        
        # 绘制AU辅助情绪柱状图
        au_bar_start_y = au_section_y + 30
        au_bar_height = 14  # 更小的高度适应更多类型
        au_bar_spacing = 3  # 更小的间距
        
        # 计算AU辅助情绪部分需要的高度
        required_au_height = len(au_display_emotion_types) * (au_bar_height + au_bar_spacing)
        if au_bar_start_y + required_au_height > content_y + content_height - 10:
            self.put_text(canvas, "空间不足", (content_x + 10, au_bar_start_y), (180, 180, 0), 14)
        else:
            # 显示序列信息
            if self.au_suggestions and self.au_suggestions.get('is_sequence', False):
                sequence_text = f"序列分析({self.au_suggestions.get('sequence_length', 0)}帧)"
                text_size = cv2.getTextSize(sequence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = content_x + content_width - text_size[0] - 10
                text_y = au_section_y + 20
                cv2.putText(canvas, sequence_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 100), 1, cv2.LINE_AA)
            
            # 为每种情绪绘制条形图
            for i, emotion_name in enumerate(au_display_emotion_types):
                # 从AU建议中获取概率值
                prob = emotion_dict.get(emotion_name, 0.0)
                if prob <= 0.001:  # 如果概率太小，显示为0
                    prob_text = "0.00"
                else:
                    prob_text = f"{prob:.2f}"
                
                # 尝试将情绪名称转换为中文显示
                try:
                    emotion_enum = getattr(EmotionType, emotion_name.upper())
                    emotion_text = str(emotion_enum)  # 显示中文名称
                except (AttributeError, ValueError):
                    emotion_text = emotion_name  # 使用英文名作为后备
                
                # 计算柱状图位置
                bar_y = au_bar_start_y + i * (au_bar_height + au_bar_spacing)
                label_x = content_x + 10
                bar_x = label_x + bar_label_width
                value_x = bar_x + bar_max_width
                
                # 显示情绪名称
                self.put_text(canvas, emotion_text, (label_x, bar_y + au_bar_height // 2 + 4), (200, 200, 200), 12)
                
                # 计算条形图宽度
                bar_width_actual = int(prob * bar_max_width)
                bar_width_actual = max(0, min(bar_width_actual, bar_max_width))
                
                # 设置条形图颜色
                try:
                    emotion_enum = getattr(EmotionType, emotion_name.upper())
                    bar_color = self.emotion_colors.get(emotion_enum, (100, 100, 100))
                except (AttributeError, ValueError):
                    bar_color = (100, 100, 100)  # 默认灰色
                
                # 绘制条形图
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + au_bar_height), (50, 50, 55), -1)
                if bar_width_actual > 0:
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width_actual, bar_y + au_bar_height), bar_color, -1)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + au_bar_height), (70, 70, 75), 1)
                
                # 显示概率值
                self.put_text(canvas, prob_text, (value_x, bar_y + au_bar_height // 2 + 4), (200, 200, 200), 12)
        
        # --- End of rendering DYNAMIC emotion content and AU auxiliary emotions ---
        
    # 删除不需要的Qt渲染方法，此方法已被上面的cv2渲染逻辑取代
        