#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
装饰器工具模块
提供单例模式装饰器等功能
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Type, TypeVar

logger = logging.getLogger(__name__)

# 单例类型变量
T = TypeVar('T')

def singleton(cls: Type[T]) -> Type[T]:
    """
    单例模式装饰器
    确保一个类只有一个实例，并提供全局访问点
    
    Args:
        cls: 需要实现单例模式的类
        
    Returns:
        装饰后的类，具有单例特性
    """
    instances: Dict[Type, Any] = {}
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper

def timing_decorator(func: Callable) -> Callable:
    """
    计时装饰器
    记录函数执行时间，并在函数执行前后输出日志
    
    Args:
        func: 需要计时的函数
        
    Returns:
        装饰后的函数，具有计时功能
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"开始执行 {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"完成执行 {func.__name__}，耗时: {elapsed:.4f}秒")
        return result
    
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    重试装饰器
    在函数执行失败时自动重试
    
    Args:
        max_attempts: 最大重试次数
        delay: 重试间隔（秒）
        
    Returns:
        装饰后的函数，具有重试功能
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"{func.__name__} 重试 {max_attempts} 次后仍然失败: {e}")
                        raise
                    logger.warning(f"{func.__name__} 执行失败，{delay}秒后进行第 {attempts+1} 次重试: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator 