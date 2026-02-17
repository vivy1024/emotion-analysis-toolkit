import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

class AUVisualizer:
    """面部动作单元(AU)可视化工具"""
    
    @staticmethod
    def visualize_aus(frame, aus, face_box=None, threshold=0.5, font_scale=0.5, thickness=1, color_inactive=(255, 0, 0), color_active=(0, 255, 0)):
        """
        可视化AU列表
        
        参数:
            frame: 输入图像
            aus: AU字典
            face_box: 人脸框
            threshold: 激活阈值，此处应为0-5范围的阈值
            font_scale: 字体大小
            thickness: 线宽
            color_inactive: 未激活的颜色
            color_active: 激活的颜色
            
        返回:
            可视化后的图像
        """
        # 深拷贝图像
        vis_img = frame.copy()
        
        # 如果没有AU，直接返回
        if not aus:
            return vis_img
        
        # 调整AU阈值为0-5范围（如果调用者未调整）
        adjusted_threshold = threshold
        if adjusted_threshold <= 1.0:
            adjusted_threshold = threshold * 5.0
        
        # 绘制AU列表
        y_offset = 30
        for i, (au_name, intensity) in enumerate(sorted(aus.items())):
            # 判断是否激活
            is_active = intensity > adjusted_threshold
            color = color_active if is_active else color_inactive
            
            # 绘制AU名称和强度
            text = f"{au_name}: {intensity:.2f}"
            cv2.putText(vis_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y_offset += 20
        
        return vis_img
    
    @staticmethod
    def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        在图像上绘制面部特征点
        
        参数:
            image: 输入图像
            landmarks: 特征点坐标数组 (n_points, 2)
            
        返回:
            带有特征点的图像
        """
        vis_img = image.copy()
        
        if landmarks is None:
            return vis_img
            
        # 确保landmarks是正确的形状
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
            
        # 绘制每个特征点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
            
        return vis_img
    
    @staticmethod
    def overlay_emotion_text(image: np.ndarray, 
                            emotions: Dict[str, float], 
                            position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        在图像上叠加情绪文本
        
        参数:
            image: 输入图像
            emotions: 情绪概率字典
            position: 文本位置 (x, y)
            
        返回:
            带有情绪文本的图像
        """
        vis_img = image.copy()
        
        if not emotions:
            return vis_img
            
        # 获取最高概率的情绪
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name, prob = top_emotion
        
        text = f"{emotion_name}: {prob:.2f}"
        cv2.putText(vis_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_img
