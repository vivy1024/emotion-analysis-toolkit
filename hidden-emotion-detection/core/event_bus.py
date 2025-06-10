#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事件总线模块
实现基于发布-订阅模式的事件驱动架构，提供模块间松耦合的通信机制
"""

import threading
import logging
import time
from typing import Dict, List, Callable, Any, Optional, Set
from queue import Queue
import traceback
from threading import Thread

from .data_types import Event, EventType

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EventBus")

class EventBus:
    """
    事件总线，实现发布-订阅模式
    支持同步和异步事件处理
    """
    
    _instance = None  # 单例实例
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化事件总线"""
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
            
        # 事件处理器字典，键为事件类型，值为处理器列表
        self.handlers = {}
        for event_type in EventType:
            self.handlers[event_type] = []
        
        # 异步事件队列
        self.event_queue = Queue()
        
        # 事件处理线程
        self.is_running = True
        self.event_thread = Thread(target=self._event_loop, daemon=True)
        self.event_thread.start()
        
        # 事件处理统计
        self._stats = {
            "published_events": 0,
            "processed_events": 0,
            "error_count": 0,
            "event_counts": {},
            "average_processing_time": {}
        }
        
        # 线程锁，保护处理器字典
        self.handlers_lock = threading.Lock()
        
        # 事件类型集合，用于验证
        self._valid_event_types: Set[EventType] = set(EventType)
        
        self._initialized = True
        logger.info("事件总线初始化完成")
    
    def start(self):
        """启动事件处理线程"""
        if self.is_running:
            logger.warning("事件总线已经在运行")
            return
            
        self.is_running = True
        logger.info("事件总线已启动")
    
    def stop(self):
        """停止事件处理线程"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.event_thread:
            self.event_thread.join(timeout=2.0)
        logger.info("事件总线已停止")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> bool:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数，接收Event参数
            
        Returns:
            bool: 订阅是否成功
        """
        if event_type not in self._valid_event_types:
            logger.error(f"无效的事件类型: {event_type}")
            return False
        
        with self.handlers_lock:
            if handler not in self.handlers[event_type]:
                self.handlers[event_type].append(handler)
                logger.debug(f"订阅事件: {event_type}, 当前订阅数: {len(self.handlers[event_type])}")
                return True
            else:
                logger.warning(f"处理器已订阅事件: {event_type}")
                return False
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> bool:
        """
        取消订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            
        Returns:
            bool: 取消订阅是否成功
        """
        if event_type not in self._valid_event_types:
            logger.error(f"无效的事件类型: {event_type}")
            return False
        
        with self.handlers_lock:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                logger.debug(f"取消订阅事件: {event_type}, 当前订阅数: {len(self.handlers[event_type])}")
                
                # 如果没有处理器，删除该事件类型
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
                    
                return True
            else:
                logger.warning(f"处理器未订阅事件: {event_type}")
                return False
    
    def publish(self, event_type: EventType, data: Any = None, source: str = "system") -> None:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
        """
        if event_type not in self._valid_event_types:
            logger.error(f"无效的事件类型: {event_type}")
            return
        
        # 更新统计信息
        self._stats["published_events"] += 1
        if event_type in self._stats["event_counts"]:
            self._stats["event_counts"][event_type] += 1
        else:
            self._stats["event_counts"][event_type] = 1
        
        event = Event(
            type=event_type,
            data=data,
            source=source,
            timestamp=int(time.time() * 1000)
        )
        
        # 将事件放入队列
        self.event_queue.put(event)
        logger.debug(f"事件已发布: {event_type}")
    
    def _event_loop(self):
        """事件处理循环"""
        logger.info("事件处理线程已启动")
        
        while self.is_running:
            try:
                # 从队列中获取事件，最多等待1秒
                try:
                    event = self.event_queue.get(timeout=1.0)
                except Exception:
                    # 队列为空，继续循环
                    continue
                
                # 处理事件
                self._handle_event(event)
                
                # 标记任务完成
                self.event_queue.task_done()
                
            except Exception as e:
                if self.is_running:  # 只有在运行时才记录错误
                    logger.error(f"事件处理线程异常: {e}")
                    traceback.print_exc()
                    self._stats["error_count"] += 1
    
    def _handle_event(self, event: Event):
        """
        处理单个事件
        
        Args:
            event: 事件对象
        """
        start_time = time.time()
        
        with self.handlers_lock:
            # 获取该事件类型的所有处理器
            handlers = self.handlers.get(event.type, [])[:]
        
        # 如果没有处理器，记录并返回
        if not handlers:
            logger.debug(f"没有处理器订阅事件: {event.type}")
            return
        
        # 调用所有处理器
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件处理器异常: {e}, 事件类型: {event.type}")
                traceback.print_exc()
                self._stats["error_count"] += 1
        
        # 更新统计信息
        self._stats["processed_events"] += 1
        processing_time = time.time() - start_time
        
        # 更新平均处理时间
        if event.type in self._stats["average_processing_time"]:
            old_avg = self._stats["average_processing_time"][event.type]
            count = self._stats["event_counts"][event.type]
            # 计算新的平均值
            new_avg = (old_avg * (count - 1) + processing_time) / count
            self._stats["average_processing_time"][event.type] = new_avg
        else:
            self._stats["average_processing_time"][event.type] = processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件总线统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        with self.handlers_lock:
            # 返回副本，避免外部修改
            return self._stats.copy()
    
    def get_queue_size(self) -> int:
        """
        获取事件队列大小
        
        Returns:
            int: 队列中事件数量
        """
        return self.event_queue.qsize()
    
    def wait_for_empty_queue(self, timeout: float = None) -> bool:
        """
        等待事件队列清空
        
        Args:
            timeout: 超时时间(秒)，None表示无限等待
            
        Returns:
            bool: 是否在超时前清空队列
        """
        try:
            self.event_queue.join(timeout=timeout)
            return True
        except Exception:
            return False
    
    def clear_stats(self):
        """清空统计信息"""
        with self.handlers_lock:
            self._stats = {
                "published_events": 0,
                "processed_events": 0,
                "error_count": 0,
                "event_counts": {},
                "average_processing_time": {}
            } 

    def shutdown(self):
        """关闭事件总线"""
        self.is_running = False
        
        # 等待事件线程结束
        if hasattr(self, 'event_thread') and self.event_thread.is_alive():
            self.event_thread.join(timeout=2.0)
        
        logger.info("事件总线已关闭") 