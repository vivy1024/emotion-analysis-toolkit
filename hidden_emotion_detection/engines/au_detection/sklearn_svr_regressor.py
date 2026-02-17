import os
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
import cv2
from sklearn.svm import SVR
from pathlib import Path
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SklearnSVRRegressor")

class SklearnAURegressor:
    """
    基于scikit-learn的AU回归器，替代OpenFace SVR实现
    """
    
    def __init__(self, models_dir: str = "enhance_hidden/models/svr_models"):
        """
        初始化回归器
        
        Args:
            models_dir: SVR模型目录
        """
        self.models_dir = models_dir
        self.au_svr_models = {}
        
        # 更新为0-5范围的阈值
        self.au_thresholds = {
            # 常见AU阈值 (0-5范围)
            'AU01': 1.5, 'AU02': 1.5, 'AU04': 1.5, 'AU05': 1.5, 
            'AU06': 1.5, 'AU07': 1.5, 'AU09': 1.5, 'AU10': 1.5,
            'AU12': 1.5, 'AU14': 1.5, 'AU15': 1.5, 'AU17': 1.5,
            'AU20': 1.5, 'AU23': 1.5, 'AU25': 1.5, 'AU26': 1.5,
            'AU28': 1.5, 'AU45': 1.5
        }
        
        # 加载SVR模型
        self._load_models()
    
    def _load_models(self):
        """加载AU检测模型"""
        
        # 检查模型目录是否存在
        if not os.path.exists(self.models_dir):
            logger.error(f"SVR模型目录不存在: {self.models_dir}")
            return
            
        logger.info(f"开始加载SVR模型，目录: {self.models_dir}")
        
        # 确认目录存在所有预期的模型文件
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl') and f.startswith('AU')]
        logger.info(f"SVR模型目录中发现{len(model_files)}个模型文件: {model_files}")
        
        # 加载所有模型
        models_loaded = 0
        for au_name in self.au_thresholds.keys():
            # 修复文件名格式: 使用AU01_svr.pkl而不是AU01.pkl
            model_path = os.path.join(self.models_dir, f"{au_name}_svr.pkl")
            
            if os.path.exists(model_path):
                try:
                    logger.info(f"加载模型 {au_name} 从路径: {model_path}")
                    model_data = joblib.load(model_path)
                    
                    # 检查模型类型，支持直接的SVR对象或包含SVR的字典
                    if isinstance(model_data, SVR):
                        # 直接使用SVR对象
                        self.au_svr_models[au_name] = model_data
                        models_loaded += 1
                        logger.info(f"模型 {au_name} 加载成功 (SVR对象)")
                    elif isinstance(model_data, dict) and 'model' in model_data and isinstance(model_data['model'], SVR):
                        # 从字典中提取SVR模型
                        self.au_svr_models[au_name] = model_data['model']
                        models_loaded += 1
                        logger.info(f"模型 {au_name} 加载成功 (从字典中提取SVR)")
                    else:
                        logger.warning(f"模型 {au_name} 格式不支持: {type(model_data)}")
                except Exception as e:
                    logger.error(f"加载模型 {au_name} 失败: {str(e)}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")
        
        # 如果未能加载任何模型，记录警告
        if models_loaded == 0:
            logger.warning("未能加载任何SVR模型，检测可能无法正常工作")
        else:
            logger.info(f"SVR模型加载完成，成功加载 {models_loaded}/{len(self.au_thresholds)} 个模型")
    
    def _create_dummy_models(self):
        """创建示例SVR模型用于测试"""
        logger.info("创建示例SVR模型用于测试")
        
        for au_name in self.au_thresholds.keys():
            # 创建SVR模型
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            
            # 简单训练数据示例 (特征维度假设为136，对应68个关键点的x,y坐标)
            X_dummy = np.random.rand(10, 136)
            y_dummy = np.random.rand(10)
            
            # 训练模型
            model.fit(X_dummy, y_dummy)
            
            # 保存模型
            self.au_svr_models[au_name] = model
            
            # 保存到文件
            model_path = os.path.join(self.models_dir, f"{au_name}_svr.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"创建并保存示例模型: {au_name}")
            except Exception as e:
                logger.error(f"无法保存模型 {au_name}: {str(e)}")
    
    def extract_features(self, face_landmarks) -> np.ndarray:
        """
        从人脸特征点提取特征
        
        Args:
            face_landmarks: 人脸68点特征点坐标 shape=(68, 2)
        
        Returns:
            提取的特征向量
        """
        if face_landmarks is None or len(face_landmarks) == 0:
            return None
            
        # 将所有特征点展平为一维向量 (简化版本)
        features = face_landmarks.reshape(-1)
        
        # 标准化特征 (可选)
        # 实际应用中可能需要更复杂的特征提取和标准化
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)
        
        return features
    
    def predict_au_intensities(self, features):
        """预测AU强度值（范围0-5），面部特征点必须是形状为(136,)的numpy数组"""
        if not isinstance(features, np.ndarray) or features.shape != (136,):
            logger.warning(f"特征应该是136维numpy数组, 但是收到了 {type(features)} 形状 {getattr(features, 'shape', 'unknown')}")
            return {}

        result = {}
        # 使用每个AU模型预测强度
        for au_name, model in self.au_svr_models.items():
            try:
                # 确保使用float类型存储预测值，避免任何隐式类型转换
                intensity = float(model.predict([features])[0])
                # 仅限制范围，不做其他处理，保留连续值
                intensity = max(0.0, min(5.0, intensity))
                result[au_name] = intensity
            except Exception as e:
                logger.error(f"预测AU {au_name} 强度失败: {e}")

        return result
    
    def get_active_aus(self, features, default_threshold=0.5):
        """获取活动的AUs列表"""
        # 预测所有AU的强度
        intensities = self.predict_au_intensities(features)
        active_aus = []

        # 确定哪些AU是活动的
        for au_name, intensity in intensities.items():
            # 获取该AU的阈值，默认使用default_threshold
            threshold = self.au_thresholds.get(au_name, default_threshold)
            
            # 如果阈值在0-1范围，需要调整为0-5范围比较
            if threshold <= 1.0:
                threshold = threshold * 5.0
                
            # 仅在强度大于阈值时认为AU是活动的
            if intensity > threshold:
                active_aus.append(au_name)

        return active_aus 