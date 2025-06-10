#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUAnalyzer: 分析AU数据的工具类
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any

class AUAnalyzer:
    """
    AU分析器类，用于分析AU数据的变化和趋势
    """
    
    def __init__(self, window_size: int = 30):
        """
        初始化AU分析器
        
        Args:
            window_size: 分析窗口大小（帧数）
        """
        self.window_size = window_size
        self.au_history = {}  # 存储每个AU的历史数据
        self.stats = {}  # 存储每个AU的统计数据
        
    def update(self, aus: Dict[str, float]) -> Dict[str, Any]:
        """
        更新AU数据并分析
        
        Args:
            aus: AU强度字典 {'AU01': 0.5, 'AU02': 0.8, ...}
            
        Returns:
            分析结果字典
        """
        if not aus:
            return {}
            
        # 初始化新的AU
        for au in aus:
            if au not in self.au_history:
                self.au_history[au] = deque(maxlen=self.window_size)
                
        # 更新历史数据
        for au, intensity in aus.items():
            self.au_history[au].append(intensity)
            
        # 分析数据
        self._analyze()
        
        return self.stats
        
    def _analyze(self):
        """分析AU历史数据"""
        for au, history in self.au_history.items():
            if not history:
                continue
                
            # 计算统计值
            mean = np.mean(history)
            std = np.std(history)
            min_val = np.min(history)
            max_val = np.max(history)
            current = history[-1] if history else 0
            
            # 计算变化率
            if len(history) > 1:
                change_rate = (current - history[-2]) / max(0.01, history[-2])
            else:
                change_rate = 0
                
            # 检测变化趋势
            trend = "stable"
            if len(history) > 5:
                recent = list(history)[-5:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    trend = "increasing"
                elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    trend = "decreasing"
            
            # 更新统计数据
            self.stats[au] = {
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "current": current,
                "change_rate": change_rate,
                "trend": trend
            }
            
    def get_active_aus(self, threshold: float = 0.2) -> List[str]:
        """
        获取当前活跃的AU列表
        
        Args:
            threshold: 活跃阈值
            
        Returns:
            活跃的AU名称列表
        """
        active = []
        for au, stats in self.stats.items():
            if stats.get("current", 0) > threshold:
                active.append(au)
        return active
        
    def get_changing_aus(self, rate_threshold: float = 0.2) -> Dict[str, float]:
        """
        获取变化率超过阈值的AU
        
        Args:
            rate_threshold: 变化率阈值
            
        Returns:
            变化率超过阈值的AU字典 {au_name: change_rate}
        """
        changing = {}
        for au, stats in self.stats.items():
            rate = stats.get("change_rate", 0)
            if abs(rate) > rate_threshold:
                changing[au] = rate
        return changing
        
    def reset(self):
        """重置分析器"""
        self.au_history.clear()
        self.stats.clear()