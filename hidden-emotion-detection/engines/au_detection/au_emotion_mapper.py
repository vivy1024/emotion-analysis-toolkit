#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUEmotionMapper: 面部动作单元与情绪的映射工具
"""

from typing import Dict, List, Tuple, Optional

class AUEmotionMapper:
    """
    将面部动作单元(AU)映射到情绪的工具类
    """
    
    def __init__(self):
        """初始化AU-情绪映射器"""
        # 情绪到AU的映射关系
        self.emotion_au_map = {
            'happy': ['AU06', 'AU12'],              # 高兴: 脸颊提升，嘴角拉伸
            'sad': ['AU01', 'AU04', 'AU15'],        # 悲伤: 内眉提升，皱眉，嘴角下拉
            'angry': ['AU04', 'AU05', 'AU07', 'AU23'], # 愤怒: 皱眉，上睑提升，眼睑紧绷，唇紧绷
            'surprise': ['AU01', 'AU02', 'AU05', 'AU26'], # 惊讶: 内眉提升，外眉提升，上睑提升，下巴下降
            'fear': ['AU01', 'AU02', 'AU04', 'AU05', 'AU20'], # 恐惧: 内眉提升，外眉提升，皱眉，上睑提升，唇部延伸
            'disgust': ['AU09', 'AU15', 'AU17'],      # 厌恶: 鼻皱，嘴角下拉，下巴提升
            'contempt': ['AU12', 'AU14'],           # 蔑视: 嘴角拉伸，酒窝
            'neutral': []                           # 中性: 无明显AU
        }
        
        # AU权重表，表示AU对特定情绪的重要程度
        self.au_weights = {
            'happy': {'AU06': 0.5, 'AU12': 0.5},
            'sad': {'AU01': 0.3, 'AU04': 0.3, 'AU15': 0.4},
            'angry': {'AU04': 0.4, 'AU05': 0.2, 'AU07': 0.2, 'AU23': 0.2},
            'surprise': {'AU01': 0.2, 'AU02': 0.2, 'AU05': 0.3, 'AU26': 0.3},
            'fear': {'AU01': 0.2, 'AU02': 0.2, 'AU04': 0.2, 'AU05': 0.2, 'AU20': 0.2},
            'disgust': {'AU09': 0.4, 'AU15': 0.3, 'AU17': 0.3},
            'contempt': {'AU12': 0.5, 'AU14': 0.5},
            'neutral': {}
        }
        
    def get_emotion_from_aus(self, aus: Dict[str, float], threshold: float = 0.2) -> Dict[str, float]:
        """
        从AU强度估计情绪概率
        
        Args:
            aus: AU强度字典 {'AU01': 0.5, 'AU02': 0.8, ...}
            threshold: AU激活阈值
            
        Returns:
            情绪概率字典 {'happy': 0.8, 'sad': 0.1, ...}
        """
        if not aus:
            return {'neutral': 1.0}
            
        # 过滤激活的AU
        active_aus = {k: v for k, v in aus.items() if v > threshold}
        
        if not active_aus:
            return {'neutral': 1.0}
            
        # 计算每种情绪的得分
        emotion_scores = {}
        for emotion, emotion_aus in self.emotion_au_map.items():
            if emotion == 'neutral':
                continue
                
            score = 0.0
            matched_aus = 0
            
            # 计算匹配的AU得分
            for au in emotion_aus:
                if au in active_aus:
                    au_weight = self.au_weights[emotion].get(au, 0.0)
                    score += active_aus[au] * au_weight
                    matched_aus += 1
                    
            # 标准化得分
            if matched_aus > 0:
                # 考虑匹配的AU数量占情绪所需AU总数的比例
                match_ratio = matched_aus / len(emotion_aus) if emotion_aus else 0
                # 综合考虑得分和匹配比例
                final_score = score * match_ratio
                emotion_scores[emotion] = final_score
                
        # 如果没有情绪得分，返回中性
        if not emotion_scores:
            return {'neutral': 1.0}
            
        # 归一化情绪得分
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
            
        # 添加中性情绪得分（1减去所有其他情绪得分的总和）
        neutral_score = max(0, 1.0 - sum(emotion_scores.values()))
        emotion_scores['neutral'] = neutral_score
            
        return emotion_scores
        
    def get_dominant_emotion(self, aus: Dict[str, float], threshold: float = 0.2) -> Tuple[str, float]:
        """
        获取最主要的情绪
        
        Args:
            aus: AU强度字典
            threshold: AU激活阈值
            
        Returns:
            (情绪名称, 概率)
        """
        emotion_scores = self.get_emotion_from_aus(aus, threshold)
        if not emotion_scores:
            return ('neutral', 1.0)
            
        return max(emotion_scores.items(), key=lambda x: x[1])
        
    def get_conflicting_emotions(self, aus: Dict[str, float], threshold: float = 0.2) -> List[Tuple[str, float]]:
        """
        获取可能存在冲突的情绪（得分接近的多种情绪）
        
        Args:
            aus: AU强度字典
            threshold: AU激活阈值
            
        Returns:
            情绪与概率元组列表，按概率降序排序
        """
        emotion_scores = self.get_emotion_from_aus(aus, threshold)
        if not emotion_scores:
            return [('neutral', 1.0)]
            
        # 按概率降序排序
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 筛选得分超过0.2的情绪
        significant_emotions = [(e, s) for e, s in sorted_emotions if s > 0.2]
        
        return significant_emotions
