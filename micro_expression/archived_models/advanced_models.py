#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级模型模块 - 提供高级模型实现

该模块提供了基于注意力机制和其他高级特性的微表情识别模型实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroExpressionAdvancedModel(nn.Module):
    """基于高级特性的微表情识别模型"""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...] = (20, 128, 128, 4),
                 output_shape: int = 7,
                 cnn_channels: Tuple[int, ...] = (32, 64, 128, 256),
                 transformer_dim: int = 256,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True,
                 use_residual: bool = True,
                 use_feature_fusion: bool = True):
        """初始化高级模型
        
        Args:
            input_shape: 输入形状，固定为 (20, 128, 128, 4)
            output_shape: 输出类别数，默认为7
            cnn_channels: CNN通道数配置
            transformer_dim: Transformer维度
            transformer_heads: Transformer注意力头数
            transformer_layers: Transformer层数
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用批归一化
            use_residual: 是否使用残差连接
            use_feature_fusion: 是否使用特征融合
        """
        super().__init__()
        
        # 验证输入形状
        if input_shape != (20, 128, 128, 4):
            logger.warning(f"输入形状 {input_shape} 不符合规范，将使用默认形状 (20, 128, 128, 4)")
            input_shape = (20, 128, 128, 4)
        
        # 保存参数
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cnn_channels = cnn_channels
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_feature_fusion = use_feature_fusion
        
        logger.info(f"初始化高级模型 - 输入形状: {input_shape}, 输出类别: {output_shape}")
        
        # CNN特征提取器
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[3]  # 输入通道数
        
        for out_channels in cnn_channels:
            block = []
            
            # 卷积层
            block.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ))
            
            # 批归一化
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
                
            # 激活函数
            block.append(nn.ReLU())
            
            # 最大池化
            block.append(nn.MaxPool2d(kernel_size=2))
            
            # Dropout
            block.append(nn.Dropout2d(dropout_rate))
            
            # 添加到模块列表
            self.cnn_layers.append(nn.Sequential(*block))
            
            # 更新输入通道数
            in_channels = out_channels
            
        # 计算CNN输出特征维度
        self.cnn_feature_size = self._get_cnn_feature_size()
        logger.info(f"CNN特征维度: {self.cnn_feature_size}")
        
        # 特征映射层
        self.feature_mapping = nn.Linear(self.cnn_feature_size, transformer_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding()
        
        # 特征融合层
        if use_feature_fusion:
            self.fusion_layer = nn.Sequential(
                nn.Linear(transformer_dim * 2, transformer_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_dim // 2, output_shape)
        )
        
    def _get_cnn_feature_size(self) -> int:
        """计算CNN输出特征维度
        
        Returns:
            特征维度大小
        """
        # 创建示例输入
        x = torch.zeros(1, self.input_shape[3], self.input_shape[1], self.input_shape[2])
        
        # 通过CNN层
        for layer in self.cnn_layers:
            x = layer(x)
            
        # 计算特征维度
        return x.view(1, -1).size(1)
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """创建位置编码
        
        Returns:
            位置编码张量
        """
        seq_len = self.input_shape[0]
        pos_encoding = torch.zeros(seq_len, self.transformer_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.transformer_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.transformer_dim))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # (1, seq_len, transformer_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, height, width, channels)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 验证输入形状
        if seq_len != self.input_shape[0] or x.size(2) != self.input_shape[1] or x.size(3) != self.input_shape[2] or x.size(4) != self.input_shape[3]:
            raise ValueError(f"输入形状 {x.shape[1:]} 与期望形状 {self.input_shape} 不匹配")
        
        # 重塑输入以便进行CNN处理
        x = x.view(batch_size * seq_len, self.input_shape[3], self.input_shape[1], self.input_shape[2])
        
        # CNN特征提取
        cnn_features = x
        for i, layer in enumerate(self.cnn_layers):
            cnn_out = layer(cnn_features)
            if self.use_residual and i > 0 and cnn_features.size() == cnn_out.size():
                cnn_features = cnn_out + cnn_features
            else:
                cnn_features = cnn_out
                
        # 重塑特征
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # 特征映射
        features = self.feature_mapping(cnn_features)
        
        # 添加位置编码
        pos_encoding = self.pos_encoding.to(features.device)
        features = features + pos_encoding
        
        # Transformer编码
        transformer_out = self.transformer_encoder(features)
        
        # 特征融合
        if self.use_feature_fusion:
            # 全局平均池化和最大池化
            avg_pool = torch.mean(transformer_out, dim=1)
            max_pool, _ = torch.max(transformer_out, dim=1)
            
            # 融合特征
            fused_features = torch.cat([avg_pool, max_pool], dim=1)
            features = self.fusion_layer(fused_features)
        else:
            features = transformer_out[:, -1]  # 取最后一个时间步的特征
            
        # 全连接层
        out = self.fc(features)
        
        return out
        
    def configure_optimizers(self, lr: float = 0.001) -> torch.optim.Optimizer:
        """配置优化器
        
        Args:
            lr: 学习率
            
        Returns:
            优化器实例
        """
        return torch.optim.Adam(self.parameters(), lr=lr) 
 
 
 