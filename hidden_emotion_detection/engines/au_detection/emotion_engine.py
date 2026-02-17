import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..au_engine import AUEngine
from .au_emotion_mapper import AUEmotionMapper

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AUEmotionEngine")

class AUEmotionEngine:
    """
    基于AU的情绪分析引擎
    使用面部动作单元(AU)预测情绪状态
    """
    
    def __init__(self, 
                 au_engine: Optional[AUEngine] = None,
                 emotion_smoothing: int = 5):
        """
        初始化AU情绪引擎
        
        参数:
            au_engine: AU引擎实例，如果为None将创建新实例
            emotion_smoothing: 情绪平滑帧数
        """
        # 创建AU引擎或使用传入的引擎
        self.au_engine = au_engine if au_engine else AUEngine(
            face_model="retinaface",
            landmark_model="mobilenet",
            au_model="svm",
            emotion_model="resmasknet",
            device="auto"
        )
        
        self.emotion_smoothing = emotion_smoothing
        
        # 情绪历史和统计
        self.emotion_history = {}
        self.emotion_stats = {}
        
        # 回调函数
        self.emotion_callback = None
        
        # 状态
        self.is_running = False
        self.latest_emotion = None
        self.latest_emotion_scores = {}
    
    def start(self, emotion_callback: Optional[Callable] = None):
        """
        启动引擎
        
        参数:
            emotion_callback: 情绪结果回调函数
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.emotion_callback = emotion_callback
        
        # 启动AU引擎
        self.au_engine.start(result_callback=self._handle_au_result)
        logger.info("AU情绪引擎已启动")
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        
        # 停止AU引擎
        self.au_engine.stop()
        logger.info("AU情绪引擎已停止")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理一帧图像
        
        参数:
            frame: 输入图像(BGR格式)
            
        返回:
            情绪分析结果
        """
        if not self.is_running:
            logger.warning("引擎未启动")
            return {}
            
        # 使用AU引擎处理帧
        au_result = self.au_engine.process_frame(frame)
        
        if not au_result or 'aus' not in au_result:
            return {}
            
        # 使用最新AU结果更新情绪
        return self._update_emotion(au_result)
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """
        获取主导情绪和置信度
        
        返回:
            (情绪名称, 置信度)元组
        """
        if not self.latest_emotion or not self.latest_emotion_scores:
            return ("neutral", 1.0)
            
        return (self.latest_emotion, self.latest_emotion_scores.get(self.latest_emotion, 0.0))
    
    def get_emotion_distribution(self) -> Dict[str, float]:
        """
        获取情绪分布
        
        返回:
            情绪概率分布字典
        """
        return self.latest_emotion_scores.copy() if self.latest_emotion_scores else {}
    
    def _handle_au_result(self, au_result: Dict):
        """
        处理AU结果，更新情绪
        
        参数:
            au_result: AU检测结果
        """
        if not au_result or 'aus' not in au_result:
            return
            
        self._update_emotion(au_result)
    
    def _update_emotion(self, au_result: Dict) -> Dict:
        """
        基于AU结果更新情绪状态
        
        参数:
            au_result: AU检测结果
            
        返回:
            情绪分析结果
        """
        # 从AU中获取激活的AU列表
        active_aus = self.au_engine.get_active_aus(au_result.get('aus', {}))
        
        # 如果没有AU激活，返回中性
        if not active_aus:
            emotion_scores = {'neutral': 1.0}
            dominant_emotion = 'neutral'
        else:
            # 获取情绪分数
            emotion_scores = AUEmotionMapper.get_emotion_from_aus(active_aus)
            dominant_emotion, confidence = AUEmotionMapper.get_dominant_emotion(active_aus)
            
            # 如果所有情绪概率都较低，考虑为中性
            if confidence < 0.3:
                emotion_scores['neutral'] = max(emotion_scores.get('neutral', 0), 0.6)
                
            # 检查是否需要平滑
            if self.emotion_smoothing > 1:
                emotion_scores = self._smooth_emotions(emotion_scores)
                
                # 重新确定主导情绪
                if emotion_scores:
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # 更新最新情绪
        self.latest_emotion = dominant_emotion
        self.latest_emotion_scores = emotion_scores
        
        # 创建结果
        emotion_result = {
            'emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'active_aus': active_aus,
            'timestamp': time.time()
        }
        
        # 调用回调
        if self.emotion_callback:
            self.emotion_callback(emotion_result)
            
        return emotion_result
    
    def _smooth_emotions(self, current_emotions: Dict[str, float]) -> Dict[str, float]:
        """
        平滑情绪得分
        
        参数:
            current_emotions: 当前情绪得分
            
        返回:
            平滑后的情绪得分
        """
        # 初始化情绪历史
        for emotion in current_emotions:
            if emotion not in self.emotion_history:
                self.emotion_history[emotion] = []
                
        # 更新情绪历史
        for emotion, score in current_emotions.items():
            history = self.emotion_history[emotion]
            history.append(score)
            
            # 保持历史限制在平滑窗口大小
            if len(history) > self.emotion_smoothing:
                history.pop(0)
        
        # 计算平滑情绪得分
        smoothed_emotions = {}
        for emotion, history in self.emotion_history.items():
            if history:
                # 加权平均，最近的得分权重更高
                weights = np.linspace(0.5, 1.0, len(history))
                smoothed_emotions[emotion] = np.average(history, weights=weights)
        
        return smoothed_emotions