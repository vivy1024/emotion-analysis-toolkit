#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型工厂模块 - 微表情识别模型工厂

该模块负责创建和管理不同类型的微表情识别模型。
支持以下模型类型：
1. cnn - 纯CNN模型，轻量快速
2. lstm - 纯LSTM模型，善于捕捉时序特征
3. cnn_lstm - CNN-LSTM混合模型，平衡空间和时序特征
4. advanced - 高级模型，使用注意力机制
5. hybrid - 混合架构，融合多种技术
"""

import logging
import torch
from typing import Dict, Any, Optional
from src.models.cnn_models import MicroExpressionModelPyTorch
from src.models.lstm_models import MicroExpressionLSTMModel
from src.models.cnn_lstm_models import MicroExpressionCNNLSTMModel
from src.models.advanced_models import MicroExpressionAdvancedModel
from src.models.hybrid_models import HybridMicroExpressionModel

# 配置日志
logger = logging.getLogger(__name__)

class ModelFactory:
    """模型工厂类"""
    
    def __init__(self, device: str = 'cuda'):
        """初始化模型工厂
        
        Args:
            device: 设备类型，可选'cuda'或'cpu'
        """
        self.device = device
        self.models = {}
        self._register_models()
        
    def _register_models(self):
        """注册所有支持的模型"""
        # 注册CNN模型
        self.register_model('cnn', MicroExpressionModelPyTorch)
        logger.info("已注册模型: cnn")
        
        # 注册LSTM模型
        self.register_model('lstm', MicroExpressionLSTMModel)
        logger.info("已注册模型: lstm")
        
        # 注册CNN-LSTM混合模型
        self.register_model('cnn_lstm', MicroExpressionCNNLSTMModel)
        logger.info("已注册模型: cnn_lstm")
        
        # 注册高级模型
        self.register_model('advanced', MicroExpressionAdvancedModel)
        logger.info("已注册模型: advanced")
        
        # 注册混合模型
        self.register_model('hybrid', HybridMicroExpressionModel)
        logger.info("已注册模型: hybrid")
        
        logger.info(f"模型工厂初始化完成，使用设备: {self.device}")
        
    def register_model(self, name: str, model_class):
        """注册模型
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        self.models[name] = model_class
        
    def create_model(self, model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
        """创建模型实例
        
        Args:
            model_type: 模型类型
            config: 模型配置
            
        Returns:
            torch.nn.Module: 模型实例
        """
        if model_type not in self.models:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        try:
            model = self.models[model_type](
                input_shape=config['input_shape'],
                output_shape=config['output_shape'],
                use_batch_norm=config.get('use_batch_norm', True),
                dropout_rate=config.get('dropout_rate', 0.3),
                use_attention=config.get('use_attention', False),
                use_residual=config.get('use_residual', False),
                use_feature_fusion=config.get('use_feature_fusion', False)
            )
            
            # 将模型移动到指定设备
            model = model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"创建模型失败: {str(e)}")
            raise
            
    def get_supported_models(self) -> Dict[str, Any]:
        """获取支持的模型列表
        
        Returns:
            Dict[str, Any]: 支持的模型列表
        """
        return {
            'cnn': '纯CNN模型，轻量快速',
            'lstm': '纯LSTM模型，善于捕捉时序特征',
            'cnn_lstm': 'CNN-LSTM混合模型，平衡空间和时序特征',
            'advanced': '高级模型，使用注意力机制',
            'hybrid': '混合架构，融合多种技术'
        } 