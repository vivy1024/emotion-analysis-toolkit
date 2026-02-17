#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 微表情面板
显示微表情分析结果，包含AU辅助情绪区域
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import traceback

from .base_panel import BasePanel
from ..core.data_types import  EmotionType, EmotionResult, AUResult
from ..core.event_bus import EventBus, Event, EventType
from .macro_emotion_panel import results_logger # <-- 导入共享的 results_logger

logger = logging.getLogger(__name__.split('.')[-1].upper())

class MicroEmotionPanel(BasePanel):
    """微表情面板，显示微表情分析结果"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化微表情面板"""
        super().__init__("微表情分析")
        self.event_bus = event_bus
        logger.info(f"MicroEmotionPanel initialized. Event bus provided: {self.event_bus is not None}")
        
        # 情绪颜色映射
        self.emotion_colors = {
            EmotionType.HAPPINESS: (0, 255, 0),    # 高兴：绿色
            EmotionType.DISGUST: (0, 140, 255),    # 厌恶：橙色
            EmotionType.SURPRISE: (255, 255, 0),   # 惊讶：青色
            EmotionType.FEAR: (255, 0, 255),       # 恐惧：紫色
            EmotionType.SADNESS: (255, 128, 0),    # 悲伤：蓝色
            EmotionType.CONFUSION: (128, 0, 128),  # 压抑：紫色
            EmotionType.ANGER: (0, 0, 255),        # 愤怒：红色
            EmotionType.CONTEMPT: (160, 0, 160),   # 蔑视：深紫色
            EmotionType.REPRESSION: (128, 0, 128), # 压抑：与CONFUSION一样的紫色
            EmotionType.NEUTRAL: (128, 128, 128),  # 中性：灰色
            EmotionType.UNKNOWN: (200, 200, 200)   # 未知：浅灰色
        }
        
        # 微表情特征描述
        self.emotion_descriptions = {
            EmotionType.HAPPINESS: "特征：嘴角轻微上扬，脸颊轻微抬起",
            EmotionType.DISGUST: "特征：鼻子轻微皱起，上唇轻微上扬",
            EmotionType.SURPRISE: "特征：眉毛轻微上扬，眼睛轻微睁大",
            EmotionType.FEAR: "特征：眉头轻微紧皱，嘴角轻微紧绷",
            EmotionType.SADNESS: "特征：内眉轻微上扬，嘴角轻微下垂",
            EmotionType.CONFUSION: "特征：嘴唇压紧/紧绷, 下巴控制",
            EmotionType.REPRESSION: "特征：面部肌肉紧绷，压抑表情",
            EmotionType.ANGER: "特征：眉头紧皱，眼睑紧绷，嘴唇紧闭",
            EmotionType.CONTEMPT: "特征：单侧嘴角抬起，嘴角酒窝",
            EmotionType.NEUTRAL: "特征：面部表情无明显变化"
        }
        
        # 当前微表情结果
        self.emotion_result: Optional[EmotionResult] = None
        self.raw_probabilities: Optional[Dict[str, float]] = None # 存储原始概率
        
        # 添加最终分析结果
        self.final_emotion_result: Optional[EmotionResult] = None
        self.final_last_update_time = 0
        
        # 添加主引擎结果
        self.raw_emotion_result: Optional[EmotionResult] = None
        self.raw_last_update_time = 0
        
        # 添加AU辅助情绪结果
        self.au_suggestions = {}  # 初始化为空字典而不是None
        self.last_au_update_time: float = 0
        
        # 新增：辅助检测状态
        self.au_assistance_active = False
        self.last_au_assistance_time = 0
        self.au_assistance_timeout = 3.0  # 3秒超时
        
        # 最后更新时间
        self.last_update_time = 0
        
        # 添加数据稳定性相关变量
        self.display_data = {}  # 当前显示的数据
        self.forced_display_interval = 1.0  # 强制显示间隔，防止闪烁
        self.min_display_time = 2.0  # 数据最小显示时间
        self.last_valid_data = None  # 上次有效数据
        self.last_valid_data_time = 0  # 上次有效数据时间
        
        # 情绪显示相关
        self.dominant_emotion = "neutral"
        self.dominant_prob = 0.0
        self.is_micro_expression = False
        self.display_timer = 0.0
        
        # 情绪概率分布
        self.emotions_probs = {emotion.name.lower(): 0.0 for emotion in EmotionType}
        self.emotions_probs["repression"] = 0.0  # 加入压抑情绪
        
        # 情绪持续显示相关
        self.min_display_time = 1.0  # 最小显示时间(秒)
        self.last_valid_data = None  # 最后一次有效数据
        self.last_valid_data_time = 0  # 最后一次有效数据时间
        self.display_data = None  # 当前显示数据
        
        # 显示控制
        self.au_suggestions = {}  # AU辅助情绪建议
        self.last_au_update_time = 0  # 上次AU更新时间
        self.forced_display_interval = 0.1  # 强制显示间隔(秒)
        
        # 辅助检测指示器状态
        self.au_assistance_active = False
        self.last_au_assistance_time = 0
        
        # --- 订阅事件 ---
        if self.event_bus:
            try:
                # 订阅三个主要事件
                self.event_bus.subscribe(EventType.RAW_MICRO_EMOTION_ANALYZED, self._on_micro_emotion_analyzed)  # 主引擎结果
                self.event_bus.subscribe(EventType.AU_MICRO_EMOTION_ANALYZED, self._on_micro_emotion_analyzed)   # AU辅助引擎结果
                self.event_bus.subscribe(EventType.MICRO_EMOTION_ANALYZED, self._on_micro_emotion_analyzed)      # 最终整合结果
                # 订阅AU辅助情绪事件
                self.event_bus.subscribe(EventType.AU_ANALYZED, self._on_au_analyzed)
                logger.info("MicroEmotionPanel subscribed to RAW_MICRO_EMOTION_ANALYZED, AU_MICRO_EMOTION_ANALYZED, MICRO_EMOTION_ANALYZED and AU_ANALYZED events.")
            except Exception as e:
                logger.error(f"Failed to subscribe MicroEmotionPanel to event: {e}", exc_info=True)
        else:
            logger.warning("No event bus provided to MicroEmotionPanel, cannot subscribe to events.")
    
    def _on_au_analyzed(self, event: Event):
        """处理AU分析事件，从AU数据推断微表情"""
        try:
            # 获取事件数据
            result_data = event.data
            if not result_data:
                logger.warning("MicroEmotionPanel: 收到空的AU分析事件数据")
                return
            
            logger.info(f"MicroEmotionPanel: 收到AU_ANALYZED事件, 数据键: {list(result_data.keys()) if isinstance(result_data, dict) else '非字典类型'}")
            
            # 是否为UI更新专用事件
            is_ui_update = result_data.get('update_ui', False)
            
            # 1. 直接从AU引擎结果获取数据
            if 'result' in result_data and isinstance(result_data['result'], AUResult):
                au_result = result_data['result']
                logger.info(f"MicroEmotionPanel: 从AU结果获取数据，AU强度: {len(au_result.au_intensities)} 项")
                
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
                    emotions_data['sequence_length'] = result_data.get('frames_count', 0)
                
                    logger.info(f"MicroEmotionPanel: 从AU强度生成情绪映射: {', '.join([f'{k}={v:.2f}' for k, v in emotions_data.items() if k not in ['is_sequence', 'sequence_length']])}")
                
                # 立即更新AU建议
                    self.au_suggestions = emotions_data.copy()
                self.last_au_update_time = time.time()
                return
                except Exception as e:
                    logger.error(f"从AU强度生成情绪映射失败: {e}", exc_info=True)
            
            # 2. 尝试从AU情绪引擎组件获取数据
            au_emotion_engine = result_data.get('components', {}).get('au_emotion_engine')
            
            # 如果直接从组件字典中无法获取 au_emotion_engine，尝试从 MicroEmotionAUEngine 获取数据
            if not au_emotion_engine and 'components' in result_data:
                micro_au_engine = result_data.get('components', {}).get('micro_emotion_au_engine')
                if micro_au_engine:
                    logger.info(f"MicroEmotionPanel: 直接从 micro_emotion_au_engine 获取数据")
                    # 直接保存 AU 辅助情绪结果，无需额外处理
                    self.au_suggestions = micro_au_engine.get_all_micro_emotions()
                    self.last_au_update_time = time.time()
                    return
            
            # 如果是UI更新专用事件，跳过时间间隔检查
            if not is_ui_update:
                # 检查强制间隔 - 如果距离上次更新时间太短，跳过以避免频繁更新
                current_time = time.time()
                if current_time - self.last_au_update_time < self.forced_display_interval:
                    return
            
            # 获取微表情辅助引擎的所有情绪建议
            if au_emotion_engine:
                emotions_data = au_emotion_engine.get_all_emotional_suggestions(is_micro=True)
                if emotions_data:
                    logger.info(f"MicroEmotionPanel: 获取到AU辅助情绪数据: {list(emotions_data.keys())}")
                else:
                    logger.debug("MicroEmotionPanel: AU引擎情绪建议为空")
                    return
            else:
                # 尝试从 result_data 中直接获取 all_probabilities
                if isinstance(result_data.get('result'), EmotionResult) and result_data.get('source') == 'AUEmotionEngine':
                    raw_probs = result_data.get('raw_probabilities')
                    if raw_probs:
                        logger.info(f"MicroEmotionPanel: 从 raw_probabilities 获取数据: {list(raw_probs.keys())}")
                        emotions_data = raw_probs.copy()
                        # 添加序列标记
                        emotions_data['is_sequence'] = result_data.get('is_sequence', False)
                        emotions_data['sequence_length'] = result_data.get('sequence_length', 0)
                    else:
                        logger.debug("MicroEmotionPanel: raw_probabilities 不可用")
                        return
                else:
                    logger.debug("MicroEmotionPanel: 无法获取 AU 辅助情绪数据")
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
                logger.info(f"MicroEmotionPanel: 更新 AU 建议数据，包含 {len(emotions_data)} 个项目")
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
    
    def _on_micro_emotion_analyzed(self, event: Event):
        """处理接收到的微表情分析事件，区分主引擎和AU辅助引擎结果"""
        try:
            valid_event_types = [
                EventType.RAW_MICRO_EMOTION_ANALYZED,  # 主引擎原始结果
                EventType.AU_MICRO_EMOTION_ANALYZED,   # AU辅助引擎结果
                EventType.MICRO_EMOTION_ANALYZED       # 最终整合结果
            ]
            
            if event.type not in valid_event_types or 'result' not in event.data:
                logger.debug(f"MicroPanel received event, but type is not valid micro emotion event or 'result' key missing. Type: {event.type}")
                return
                
            result_data = event.data['result']
            source = event.data.get('source', 'main_engine') # Default source if missing
            
            logger.debug(f"MicroPanel received {event.type} event from {source}. Data keys: {list(event.data.keys()) if isinstance(event.data, dict) else 'Not a dict'}")
            
            if not isinstance(result_data, EmotionResult):
                logger.warning(f"MicroEmotionPanel: 收到{event.type}事件，但结果数据不是EmotionResult类型: {type(result_data)}")
                return
            
            # 检查是否是微表情结果 (对于RAW_MICRO_EMOTION_ANALYZED和AU_MICRO_EMOTION_ANALYZED都应该是)
            if getattr(result_data, 'is_micro_expression', False) or event.type == EventType.RAW_MICRO_EMOTION_ANALYZED:
                # 基于事件类型处理
                if event.type == EventType.RAW_MICRO_EMOTION_ANALYZED:
                    # 处理主引擎结果
                    logger.debug(f"MicroPanel received result from main engine: {result_data.emotion_type} ({result_data.probability:.4f})")

                    # 保存主引擎结果供显示
                    self.raw_emotion_result = result_data
                    self.raw_last_update_time = time.time()
                    
                    self.emotion_result = result_data
                    # 删除可能存在的AU引擎标记
                    if hasattr(self.emotion_result, 'from_au_engine'):
                        delattr(self.emotion_result, 'from_au_engine')
                        
                    # 保存原始概率
                    self.raw_probabilities = event.data.get('raw_probabilities')
                    if self.raw_probabilities:
                        logger.debug(f"MicroPanel stored raw probabilities: {self.raw_probabilities}")
                    else:
                        logger.warning("MicroPanel received main engine event but raw_probabilities missing.")
                        self.raw_probabilities = None # 确保清空旧的
                        
                    # 记录到专用日志
                    results_logger.info(f"MICRO_RAW - Emotion: {str(self.emotion_result.emotion_type)}, Probability: {self.emotion_result.probability:.4f}")
                    self.last_update_time = time.time() # 更新主引擎结果时间戳
                    logger.debug(f"MicroEmotionPanel updated with main result: {str(self.emotion_result.emotion_type)} ({self.emotion_result.probability:.4f})")
                    
                elif event.type == EventType.AU_MICRO_EMOTION_ANALYZED:
                    # 处理AU辅助引擎结果
                    logger.debug(f"MicroPanel received AU_MICRO suggestion: {result_data.emotion_type} ({result_data.probability:.2f})")
                    
                    # 定义允许使用后备结果的超时时间
                    fallback_timeout = 1.0 
                    
                    # 条件：没有当前结果，或者距离上次主引擎更新已超过超时时间
                    use_fallback = (self.emotion_result is None or 
                                    (time.time() - self.last_update_time) > fallback_timeout)
                                    
                    if use_fallback:
                        self.emotion_result = result_data # 使用 AU 建议更新显示
                        # 标记为来自AU引擎
                        setattr(self.emotion_result, 'from_au_engine', True)
                        self.raw_probabilities = None # AU 建议不提供原始概率，清空
                        # 记录 AU 建议到日志
                        results_logger.info(f"MICRO_AU - Emotion: {str(self.emotion_result.emotion_type)}, Probability: {self.emotion_result.probability:.4f} (from AU)")
                        logger.info(f"MicroEmotionPanel updated with AU fallback result: {str(self.emotion_result.emotion_type)} ({self.emotion_result.probability:.2f})")
                    else:
                        logger.debug(f"Ignoring AU suggestion for micro panel because a recent main result exists (last update: {time.time() - self.last_update_time:.2f}s ago).")
                
                elif event.type == EventType.MICRO_EMOTION_ANALYZED:
                    # 处理最终整合结果 (始终接受)
                    logger.debug(f"MicroPanel received FINAL micro result: {result_data.emotion_type} ({result_data.probability:.4f})")
                    
                    # 保存最终分析结果
                    self.final_emotion_result = result_data
                    self.final_last_update_time = time.time()
                    
                    # 同时更新通用结果
                    self.emotion_result = result_data
                    
                    # 原始概率可能不存在
                    self.raw_probabilities = event.data.get('raw_probabilities', None)
                    
                    # 记录到专用日志
                    results_logger.info(f"MICRO_FINAL - Emotion: {str(self.emotion_result.emotion_type)}, Probability: {self.emotion_result.probability:.4f}")
                    self.last_update_time = time.time()
                
            else:
                logger.debug(f"Received {event.type} event, but result is not marked as micro-expression")

        except Exception as e:
            logger.error(f"Error processing micro emotion event in MicroEmotionPanel: {e}", exc_info=True)
    
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上。先使用 BasePanel 的 render 绘制框架，再绘制动态微表情内容。
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
            # logger is not defined in this scope, use logging directly
            logging.error(f"Error during BasePanel render call from MicroEmotionPanel: {e}")
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
        model_name = "微表情检测"
        source_text = f"来源: 微表情主引擎"
        self.put_text(canvas, source_text, (content_x + 10, content_y + 20), (150, 150, 220), 14)
            
        content_y += 30  # 下移到主引擎区域下方
        
        bar_label_width = 55 # Example value, adjust as needed
        bar_value_width = 40 # Example value, adjust as needed
        bar_container_width = content_width - bar_label_width - bar_value_width - 20 # Adjust spacing
        bar_height = 16  # 减小以适应更多情绪类型

        # 如果没有检测结果，显示空白区域，不显示任何文字
        if self.emotion_result is None:
            cv2.rectangle(canvas, (content_x, content_y), (content_x + content_width, content_y + 30), (35, 35, 40), -1)
            return
            
        # --- Code below this point will only execute if self.emotion_result is NOT None ---
        
        # --- 使用存储的原始概率，如果没有则显示错误或默认值 ---
        current_raw_probs = self.raw_probabilities
        
        # 定义期望显示的情绪类型列表 (与柱状图一致)
        display_emotion_types = [EmotionType.HAPPINESS, EmotionType.DISGUST, EmotionType.SURPRISE, EmotionType.CONFUSION, EmotionType.NEUTRAL] # 通常不显示 FEAR 和 SADNESS，因为模型不直接预测它们
        # 如果需要显示 Fear 和 Sadness，可以添加到列表中，它们的概率会是 0
        # display_emotion_types = [EmotionType.HAPPINESS, EmotionType.DISGUST, EmotionType.SURPRISE, EmotionType.FEAR, EmotionType.SADNESS, EmotionType.CONFUSION, EmotionType.NEUTRAL]
        
        # 检查原始概率是否存在且是字典
        if not isinstance(current_raw_probs, dict):
            logger.warning("Raw probabilities not available or not a dictionary in render.")
            # 可以选择显示错误信息或默认值 (全0)
            current_raw_probs = {etype.name.lower(): 0.0 for etype in display_emotion_types}
            # 或者直接返回? 但我们已经有了 dominant_result, 也许只显示主导文本更好？
            # 为了保持一致性，我们还是继续绘制柱状图，但值都是0

        # 提取主导情绪（仍然来自 self.emotion_result）
        dominant_emotion = self.emotion_result.emotion_type
        dominant_prob = self.emotion_result.probability # 这是平滑/阈值后的概率
        
        # 显示主导微表情文本 (保持不变)
        # 检查是否来自AU引擎
        source_indicator = ""
        if hasattr(self.emotion_result, 'from_au_engine') and getattr(self.emotion_result, 'from_au_engine', False):
            source_indicator = " (AU辅助)"
            
        dominant_text = f"当前微表情: {str(dominant_emotion)}{source_indicator} ({dominant_prob:.2f})"
        self.put_text(canvas, dominant_text, (content_x + 10, content_y + 20), (0, 255, 255), 16) # 显示为青色
                
        # 绘制所有微表情的概率柱状图 (现在使用 current_raw_probs)
        bar_start_y = content_y + 40 # Start slightly lower than before
        
        # --- 调试：打印当前用于绘图的原始概率 ---
        logger.debug(f"Rendering bar chart with raw_probs: {current_raw_probs}")
        # ---------------------------------------
        
        required_bar_height = len(display_emotion_types) * (bar_height + 4) # 减小间距
        if bar_start_y + required_bar_height > content_y + content_height / 2 - 20: # 给AU辅助部分预留空间
            self.put_text(canvas, "空间不足", (content_x + 10, bar_start_y), (180, 180, 0), 14)
        else:
            # 使用 display_emotion_types 列表来控制顺序和包含的类别
            for i, emotion_type in enumerate(display_emotion_types):
                emotion_text = str(emotion_type) # 中文名称
                # 从 current_raw_probs 获取原始概率，注意键是小写英文名
                prob = current_raw_probs.get(emotion_type.name.lower(), 0.0)
                prob_text = f"{prob:.2f}"
                
                # 计算柱状图位置 (absolute coordinates)
                bar_y = bar_start_y + i * (bar_height + 4) # 减小间距
                label_x = content_x + 10
                bar_x = label_x + bar_label_width
                value_x = bar_x + bar_container_width
                
                self.put_text(canvas, emotion_text, (label_x, bar_y + bar_height // 2 + 4), (200, 200, 200), 13)
                
                # 使用原始概率计算柱状图宽度
                bar_width_actual = int(prob * bar_container_width)
                bar_width_actual = max(0, min(bar_width_actual, bar_container_width))
                
                # 高亮主导情绪 (基于 self.emotion_result 中的 dominant_emotion)
                bar_color = (255, 100, 100) if emotion_type == dominant_emotion else (100, 100, 100) 
                border_color = (255, 255, 255) if emotion_type == dominant_emotion else (70, 70, 75)
                
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_container_width, bar_y + bar_height), (60, 60, 65), -1)
                if bar_width_actual > 0:
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width_actual, bar_y + bar_height), bar_color, -1)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_container_width, bar_y + bar_height), border_color, 1)
                
                self.put_text(canvas, prob_text, (value_x, bar_y + bar_height // 2 + 4), (200, 200, 200), 12)
        
        # --- 添加AU辅助情绪部分 ---
        # 计算AU辅助部分的位置
        au_section_y = content_y + content_height // 2
        
        # 绘制分隔线和背景框
        cv2.rectangle(canvas, 
                    (content_x, au_section_y), 
                    (content_x + content_width, content_y + content_height - 5), 
                    (40, 40, 50), -1)
        
        # 添加AU辅助数据来源指示 - 使用与上方相同的字体大小
        au_source = "来源: AU辅助待机"
        self.put_text(canvas, au_source, (content_x + 10, au_section_y + 20), (150, 150, 220), 14)
        
        # 检查AU辅助状态是否过期
        if self.au_assistance_active and time.time() - self.last_au_assistance_time > self.au_assistance_timeout:
            self.au_assistance_active = False
        
        # 如果没有AU辅助情绪数据，显示空白区域，不再显示等待文字
        if not self.au_suggestions or time.time() - self.last_au_update_time > 3.0:
            # 创建空的情绪字典，保证后续代码有数据可用
            emotion_dict = {}
            # 提前返回，不再处理AU辅助情绪部分的渲染
            return
        else:
            # 过滤掉非情绪类型的键
            emotion_dict = {k: v for k, v in self.au_suggestions.items() 
                          if k not in ['is_sequence', 'sequence_length']}
        
        # 定义完整FACS系统的微表情辅助情绪显示顺序
        # FACS系统情绪：Happiness, Sadness, Surprise, Fear, Anger, Disgust, Contempt, Repression
        au_display_emotion_types = [
            "happiness", "disgust", "surprise", "fear", "sadness", 
            "anger", "contempt", "repression"
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
                text_x = content_x + content_width // 2 - text_size[0] // 2
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
                value_x = bar_x + bar_container_width
                
                # 显示情绪名称
                self.put_text(canvas, emotion_text, (label_x, bar_y + au_bar_height // 2 + 4), (200, 200, 200), 12)
                
                # 计算条形图宽度
                bar_width_actual = int(prob * bar_container_width)
                bar_width_actual = max(0, min(bar_width_actual, bar_container_width))
                
                # 设置条形图颜色
                try:
                    emotion_enum = getattr(EmotionType, emotion_name.upper())
                    bar_color = self.emotion_colors.get(emotion_enum, (100, 100, 100))
                    # 略微增强颜色亮度，提高可见性
                    bar_color = tuple(min(255, int(c * 1.2)) for c in bar_color)
                except (AttributeError, ValueError):
                    bar_color = (100, 100, 100)  # 默认灰色
                
                # 绘制条形图
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_container_width, bar_y + au_bar_height), (50, 50, 55), -1)
                if bar_width_actual > 0:
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width_actual, bar_y + au_bar_height), bar_color, -1)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_container_width, bar_y + au_bar_height), (70, 70, 75), 1)
                
                # 显示概率值
                self.put_text(canvas, prob_text, (value_x, bar_y + au_bar_height // 2 + 4), (200, 200, 200), 12)