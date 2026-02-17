#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏表情检测系统 - 核心基础组件
包含系统中各种基础数据结构和接口定义
"""

import enum
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Tuple, Union

logger = logging.getLogger("core.base")

# 情绪类型枚举
class EmotionType(enum.Enum):
    """表情情绪类型枚举"""
    NEUTRAL = "中性"
    HAPPINESS = "高兴"
    SADNESS = "悲伤"
    ANGER = "愤怒"
    FEAR = "恐惧"
    SURPRISE = "惊讶"
    DISGUST = "厌恶"
    
    UNKNOWN = "未知"

# 事件类型枚举
class EventType(enum.Enum):
    """系统事件类型枚举"""
    FACE_DETECTED = "face_detected"           # 人脸检测事件
    MACRO_RESULT = "macro_expression_result"  # 宏观表情结果事件
    MICRO_RESULT = "micro_expression_result"  # 微表情结果事件
    ACTION_UNIT_RESULT = "action_unit_result" # 动作单元结果事件
    HIDDEN_EMOTION_DETECTED = "hidden_emotion_detected" # 隐藏表情检测事件
    SYSTEM_ERROR = "system_error"             # 系统错误事件

@dataclass
class Point:
    """二维点"""
    x: int = 0
    y: int = 0
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def as_tuple(self) -> Tuple[int, int]:
        """转换为元组表示"""
        return (self.x, self.y)

@dataclass
class FaceDetection:
    """人脸检测结果"""
    face_id: int = 0                   # 人脸ID
    x: int = 0                         # 左上角X坐标
    y: int = 0                         # 左上角Y坐标
    width: int = 0                     # 宽度
    height: int = 0                    # 高度
    confidence: float = 0.0            # 置信度
    landmarks: List[Tuple[int, int]] = field(default_factory=list)  # 人脸关键点
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 旋转角度(pitch, yaw, roll)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # 时间戳

@dataclass
class EmotionResult:
    """情绪分析结果基类"""
    emotion: EmotionType = EmotionType.NEUTRAL  # 情绪类型
    confidence: float = 0.0                     # 置信度
    all_scores: Dict[EmotionType, float] = field(default_factory=dict)  # 所有情绪得分

@dataclass
class MacroEmotionResult(EmotionResult):
    """宏观表情分析结果"""
    pass

@dataclass
class MicroEmotionResult(EmotionResult):
    """微表情分析结果"""
    start_time: int = 0          # 开始时间戳
    duration: int = 0            # 持续时间(毫秒)
    intensity: float = 0.0       # 强度

@dataclass
class ActionUnit:
    """动作单元"""
    au_id: int = 0               # AU ID
    name: str = ""               # AU名称
    description: str = ""        # AU描述
    intensity: float = 0.0       # 强度(0-5)
    is_active: bool = False      # 是否激活

@dataclass
class ActionUnitResult:
    """动作单元分析结果"""
    action_units: Dict[int, ActionUnit] = field(default_factory=dict)  # AU字典
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # 时间戳

@dataclass
class HiddenEmotionResult:
    """隐藏表情分析结果"""
    detected: bool = False                      # 是否检测到隐藏表情
    visible_emotion: EmotionType = EmotionType.NEUTRAL  # 表面情绪
    hidden_emotion: EmotionType = EmotionType.NEUTRAL   # 隐藏情绪
    confidence: float = 0.0                     # 置信度
    evidence: List[str] = field(default_factory=list)   # 证据线索
    au_conflicts: List[Tuple[int, int]] = field(default_factory=list)  # AU冲突

@dataclass
class FrameResult:
    """单帧分析结果"""
    frame_id: int = 0            # 帧ID
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # 时间戳
    face_detection: Optional[FaceDetection] = None  # 人脸检测结果
    macro_emotion: Optional[MacroEmotionResult] = None  # 宏观表情结果
    micro_emotion: Optional[MicroEmotionResult] = None  # 微表情结果
    au_result: Optional[ActionUnitResult] = None  # AU分析结果
    hidden_emotion: Optional[HiddenEmotionResult] = None  # 隐藏表情结果

@dataclass
class Event:
    """系统事件"""
    type: EventType              # 事件类型
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # 时间戳
    data: Dict[str, Any] = field(default_factory=dict)  # 事件数据

class EventBus:
    """事件总线，用于系统组件间通信"""
    
    def __init__(self):
        self._subscribers = {}  # 订阅者字典
        self._event_queue = queue.Queue()  # 事件队列
        self._running = False  # 运行标志
        self._worker_thread = None  # 工作线程
        self._lock = threading.Lock()  # 线程锁
    
    def start(self):
        """启动事件总线"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
            logger.info("事件总线已启动")
    
    def stop(self):
        """停止事件总线"""
        with self._lock:
            if not self._running:
                return
                
            self._running = False
            
            # 清空队列并放入停止信号
            with self._event_queue.mutex:
                self._event_queue.queue.clear()
            
            self._event_queue.put(None)  # 放入停止信号
            
            # 等待工作线程结束
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
                
            logger.info("事件总线已停止")
    
    def _process_events(self):
        """事件处理线程"""
        while self._running:
            try:
                # 从队列获取事件
                event = self._event_queue.get(timeout=0.1)
                
                # 检查停止信号
                if event is None:
                    break
                    
                # 处理事件
                self._dispatch_event(event)
                
                # 标记任务完成
                self._event_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"事件处理错误: {e}")
                
        logger.info("事件处理线程已退出")
    
    def _dispatch_event(self, event: Event):
        """分发事件到订阅者"""
        event_type = event.type
        
        # 获取该事件类型的所有订阅者
        subscribers = self._subscribers.get(event_type, [])
        
        # 调用每个订阅者的处理函数
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"事件处理器错误: {e}, 事件类型: {event_type}")
    
    def publish(self, event: Event):
        """发布事件"""
        if not self._running:
            logger.warning("事件总线未启动，无法发布事件")
            return
            
        self._event_queue.put(event)
    
    def publish_event(self, event_type: EventType, data: Dict[str, Any] = None):
        """发布指定类型的事件"""
        if data is None:
            data = {}
            
        event = Event(type=event_type, data=data)
        self.publish(event)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """订阅事件"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
                
            self._subscribers[event_type].append(callback)
            logger.debug(f"订阅事件: {event_type}, 当前订阅者数量: {len(self._subscribers[event_type])}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """取消订阅事件"""
        with self._lock:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"取消订阅事件: {event_type}, 当前订阅者数量: {len(self._subscribers[event_type])}")
    
    def wait_for_empty_queue(self, timeout=None):
        """等待事件队列处理完毕"""
        if not self._running:
            return
            
        try:
            self._event_queue.join(timeout=timeout)
        except Exception:
            pass

class IEngine:
    """分析引擎接口"""
    
    def init(self) -> bool:
        """初始化引擎"""
        raise NotImplementedError
    
    def process(self, data: Any) -> Any:
        """处理数据"""
        raise NotImplementedError
    
    def shutdown(self) -> None:
        """关闭引擎"""
        raise NotImplementedError 