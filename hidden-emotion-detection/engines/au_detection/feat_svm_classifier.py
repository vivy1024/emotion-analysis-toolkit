#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import joblib
import pickle
import logging
import cv2
from glob import glob
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.svm import LinearSVC

logger = logging.getLogger("AU_SVM_Classifier")


class SklearnSVMClassifier:
    """
    使用预训练py-feat SVM模型进行AU检测的分类器
    """
    
    def __init__(self, model_dir: str = "enhance_hidden/models/svm_models") -> None:
        """
        初始化SVM分类器

        Args:
            model_dir: SVM模型文件目录
        """
        self.model_dir = model_dir
        self.au_models: Dict[str, Any] = {}
        self.au_scalers: Dict[str, Any] = {}
        self.au_names: List[str] = []
        self.feature_dimensions: Dict[str, int] = {}  # 存储每个模型的特征维度
        self.hog_config_cache: Dict[int, Dict] = {}   # 缓存不同维度的HOG配置
        self.feature_cache: Dict = {}  # 特征缓存
        
        # 预定义的维度配置，基于已知的训练模型配置
        self.predefined_configs = {
            912: {"orientations": 8, "pixels_per_cell": (12, 12), "cells_per_block": (2, 2)},
            1195: {"orientations": 9, "pixels_per_cell": (10, 10), "cells_per_block": (2, 2)},
            1379: {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
        }
        
        # 启用调试模式(1次检测保存特征图像)
        self.debug_mode = True
        self.debug_counter = 0
        self.debug_dir = "logs/au_debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 加载主要SVM模型
        self._load_models()
        
        # 计算最常用的目标维度
        self._calculate_common_dimensions()
        
    def _calculate_common_dimensions(self) -> None:
        """计算最常见的特征维度并为其优化HOG参数"""
        if not self.feature_dimensions:
            return
            
        # 统计各维度出现频率
        dim_counts = {}
        for dim in self.feature_dimensions.values():
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
            
        # 找出最常见的维度
        self.common_dimensions = sorted(dim_counts.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"模型使用的维度统计: {self.common_dimensions}")
        
        # 为每个常用维度预先计算最佳HOG配置
        for dim, count in self.common_dimensions:
            self._find_best_hog_config(dim)
            
    def _find_best_hog_config(self, target_dim: int) -> Dict:
        """为目标维度找到最佳HOG配置参数"""
        # 首先检查是否有预定义配置
        if target_dim in self.predefined_configs:
            logger.info(f"使用预定义HOG参数配置，目标维度: {target_dim}")
            self.hog_config_cache[target_dim] = self.predefined_configs[target_dim]
            return self.predefined_configs[target_dim]
            
        # 如果已有缓存结果，直接返回
        if target_dim in self.hog_config_cache:
            return self.hog_config_cache[target_dim]

        # HOG参数组合尝试列表 (更全面)
        hog_configs = []
        
        # 生成参数搜索空间
        for orientations in [8, 9, 10, 12]:
            for ppc_size in [4, 6, 8, 10, 12, 16]:
                for cpb_size in [2, 3, 4]:
                    # 计算理论特征维度
                    ppc = (ppc_size, ppc_size)
                    cpb = (cpb_size, cpb_size)
                    
                    n_cells_y = max(0, (96 // ppc[0]) - (cpb[0] - 1))
                    n_cells_x = max(0, (96 // ppc[1]) - (cpb[1] - 1))
                    
                    if n_cells_y <= 0 or n_cells_x <= 0:
                        continue
                        
                    feature_dim = orientations * cpb[0] * cpb[1] * n_cells_y * n_cells_x
                    
                    # 存储配置和对应维度
                    hog_configs.append({
                        "orientations": orientations,
                        "pixels_per_cell": ppc,
                        "cells_per_block": cpb,
                        "estimated_dim": feature_dim,
                        "diff": abs(feature_dim - target_dim)
                    })
        
        # 按与目标维度的差异排序
        hog_configs.sort(key=lambda x: x["diff"])
        
        # 找到最接近目标维度的配置
        if hog_configs:
            best_config = {
                "orientations": hog_configs[0]["orientations"],
                "pixels_per_cell": hog_configs[0]["pixels_per_cell"],
                "cells_per_block": hog_configs[0]["cells_per_block"]
            }
            
            logger.info(f"为维度 {target_dim} 找到最佳HOG配置: {best_config}，估计维度: {hog_configs[0]['estimated_dim']}")
            
            # 缓存结果
            self.hog_config_cache[target_dim] = best_config
            return best_config
        else:
            # 默认配置
            default_config = {
                "orientations": 8,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (3, 3)
            }
            self.hog_config_cache[target_dim] = default_config
            return default_config
        
    def _load_models(self) -> None:
        """加载SVM模型和对应的标准化器"""
        try:
            # 主模型文件
            main_model_path = os.path.join(self.model_dir, "svm_model.pkl")
            if os.path.exists(main_model_path):
                with open(main_model_path, 'rb') as f:
                    self.au_models = pickle.load(f)
                
                # 提取支持的AU名称
                self.au_names = [key for key in self.au_models.keys()]
                logger.info(f"已加载主模型，支持的AU: {self.au_names}")
                
                # 获取每个模型的特征维度
                for au_name, model in self.au_models.items():
                    if hasattr(model, 'coef_'):
                        self.feature_dimensions[au_name] = model.coef_.shape[1]
                        logger.info(f"{au_name} 模型期望特征维度: {self.feature_dimensions[au_name]}")
            else:
                # 没有主模型文件，尝试加载单独的模型文件
                model_files = glob(os.path.join(self.model_dir, "AU*_svm.pkl"))
                for model_file in model_files:
                    au_name = os.path.basename(model_file).split("_")[0]
                    try:
                        model = joblib.load(model_file)
                        self.au_models[au_name] = model
                        self.au_names.append(au_name)
                        
                        # 获取模型的特征维度
                        if hasattr(model, 'coef_'):
                            self.feature_dimensions[au_name] = model.coef_.shape[1]
                            logger.info(f"{au_name} 模型期望特征维度: {self.feature_dimensions[au_name]}")
                    except Exception as e:
                        logger.error(f"加载模型 {model_file} 失败: {e}")
                
                logger.info(f"已加载 {len(self.au_models)} 个模型")
                
            # 加载标准化器
            scaler_files = glob(os.path.join(self.model_dir, "*scaler*.pkl"))
            for scaler_file in scaler_files:
                au_name = os.path.basename(scaler_file).split("_")[0]
                try:
                    self.au_scalers[au_name] = joblib.load(scaler_file)
                except Exception as e:
                    logger.error(f"加载标准化器 {scaler_file} 失败: {e}")
            
            logger.info(f"已加载 {len(self.au_scalers)} 个标准化器")
            
        except Exception as e:
            logger.error(f"加载SVM模型失败: {e}")
            raise
    
    def extract_hog_features(self, face_image: np.ndarray, target_dimension: int = None) -> np.ndarray:
        """
        从面部图像中提取HOG特征，并根据目标维度调整特征
        
        Args:
            face_image: 人脸图像
            target_dimension: 目标特征维度，如果提供则调整特征维度
            
        Returns:
            提取的HOG特征向量
        """
        # 生成缓存键
        if target_dimension is None:
            cache_key = "default"
        else:
            cache_key = str(target_dimension)
            
        # 检查特征缓存 - 禁用缓存尝试修复
        # if hasattr(face_image, 'shape') and face_image is not None:
        #     image_hash = hash(face_image.tobytes())
        #     cache_key = f"{image_hash}_{cache_key}"
        #     
        #     if cache_key in self.feature_cache:
        #         return self.feature_cache[cache_key]
        
        # 确保图像格式正确
        if face_image.dtype != np.float32:
            img = face_image.astype(np.float32)
        else:
            img = face_image.copy()
        
        # 确保图像尺寸一致
        img = cv2.resize(img, (96, 96))  # py-feat使用的尺寸
        
        # 保存一份调试图像
        if self.debug_mode and self.debug_counter < 5:
            debug_img_path = os.path.join(self.debug_dir, f"face_input_{self.debug_counter}.jpg")
            cv2.imwrite(debug_img_path, img)
            self.debug_counter += 1
            logger.info(f"已保存调试图像: {debug_img_path}")
            
        # 使用scikit-image提取HOG特征
        from skimage.feature import hog
        
        # 根据目标维度选择HOG参数
        if target_dimension:
            # 使用预计算的最佳HOG配置
            hog_config = self._find_best_hog_config(target_dimension)
            
            logger.info(f"使用HOG配置: {hog_config} 提取特征，目标维度: {target_dimension}")
            
            try:
                features, hog_image = hog(
                    img, 
                    orientations=hog_config["orientations"],
                    pixels_per_cell=hog_config["pixels_per_cell"],
                    cells_per_block=hog_config["cells_per_block"],
                    visualize=True,
                    channel_axis=2 if len(img.shape) > 2 else None
                )
                
                # 保存HOG特征可视化图像
                if self.debug_mode and self.debug_counter <= 5:
                    # 归一化HOG图像以便于显示
                    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
                    debug_hog_path = os.path.join(self.debug_dir, f"hog_visual_{target_dimension}_{self.debug_counter}.jpg")
                    cv2.imwrite(debug_hog_path, hog_image * 255)
                    logger.info(f"已保存HOG特征图: {debug_hog_path}, 特征维度: {len(features)}")
                    
            except Exception as e:
                logger.error(f"HOG特征提取失败: {e}")
                # 降级到默认参数
                features, _ = hog(
                    img, 
                    orientations=8, 
                    pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), 
                    visualize=True,
                    channel_axis=2 if len(img.shape) > 2 else None
                )
        else:
            # 使用默认参数
            features, _ = hog(
                img, 
                orientations=8, 
                pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), 
                visualize=True,
                channel_axis=2 if len(img.shape) > 2 else None
            )
        
        # 确保特征维度匹配
        if target_dimension and len(features) != target_dimension:
            # 记录原始维度
            original_dim = len(features)
            
            if len(features) > target_dimension:
                # 如果特征维度过大，截断
                features = features[:target_dimension]
                logger.warning(f"特征维度不匹配! 原始: {original_dim}，目标: {target_dimension}，已截断")
            else:
                # 如果特征维度过小，补零
                padding = np.zeros(target_dimension - len(features))
                features = np.concatenate([features, padding])
                logger.warning(f"特征维度不匹配! 原始: {original_dim}，目标: {target_dimension}，已补零")
        
        # 存入缓存 - 临时禁用缓存
        result = features.reshape(1, -1)
        # if cache_key:
        #     self.feature_cache[cache_key] = result
        #     
        #     # 控制缓存大小
        #     if len(self.feature_cache) > 100:  # 最多缓存100个特征
        #         # 删除最早添加的项
        #         old_keys = list(self.feature_cache.keys())[:50]
        #         for k in old_keys:
        #             del self.feature_cache[k]
                    
        return result
        
    def detect(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        从人脸图像中检测动作单元
        
        Args:
            face_image: 人脸图像
            
        Returns:
            包含各AU检测分数的字典
        """
        if len(self.au_models) == 0:
            raise ValueError("未加载任何AU模型，无法进行检测")
        
        try:
            # 对每个AU进行预测
            au_results = {}
            active_au_count = 0
            
            # 收集相同维度的AU模型进行批量处理
            dimension_groups = {}
            for au_name, model in self.au_models.items():
                if au_name in self.feature_dimensions:
                    dim = self.feature_dimensions[au_name]
                    if dim not in dimension_groups:
                        dimension_groups[dim] = []
                    dimension_groups[dim].append(au_name)
            
            logger.info(f"共有 {len(dimension_groups)} 个不同维度的AU模型组")
            
            # 对每个维度组只提取一次特征
            for dim, au_group in dimension_groups.items():
                # 为该维度提取一次特征
                features = self.extract_hog_features(face_image, dim)
                logger.info(f"维度 {dim} 提取特征完成，特征形状: {features.shape}")
                
                # 处理该维度下的所有AU
                for au_name in au_group:
                    model = self.au_models[au_name]
                    
                    # 尝试标准化特征
                    feat = features
                    if au_name in self.au_scalers:
                        try:
                            feat = self.au_scalers[au_name].transform(features)
                        except Exception as e:
                            logger.debug(f"特征标准化失败: {e}")
                    
                    # 获取预测结果
                    try:
                        prediction = model.predict(feat)[0]
                        # 获取决策函数值作为置信度
                        confidence = model.decision_function(feat)[0]
                        
                        # 存储结果，格式为AU01, AU02等
                        au_num = au_name.replace("AU", "")
                        au_key = f"AU{au_num.zfill(2)}" if len(au_num) == 1 else f"AU{au_num}"
                        au_results[au_key] = float(confidence)
                        
                        # 记录预测结果
                        if confidence > 0:
                            active_au_count += 1
                            logger.info(f"{au_key} 检测结果: {confidence:.4f}, 二进制预测: {prediction}")
                        
                    except Exception as e:
                        logger.error(f"{au_name} 预测失败: {e}")
                        # 如果预测失败，设为0
                        au_num = au_name.replace("AU", "")
                        au_key = f"AU{au_num.zfill(2)}" if len(au_num) == 1 else f"AU{au_num}"
                        au_results[au_key] = 0.0
            
            # 报告检测结果
            logger.info(f"AU检测完成，共检测到 {active_au_count} 个激活的AU (总计 {len(au_results)} 个AU)")
            return au_results
        
        except Exception as e:
            logger.error(f"AU检测失败: {e}")
            return {}


# 用于测试
if __name__ == "__main__":
    import cv2
    # 初始化检测器
    detector = SklearnSVMClassifier()
    
    # 创建一个测试图像
    test_image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)
    
    # 测试检测
    results = detector.detect(test_image)
    print(f"检测结果: {results}") 