#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN-LSTM混合模型模块 - 提供CNN-LSTM混合模型实现

该模块提供了基于CNN和LSTM的混合微表情识别模型实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroExpressionCNNLSTMModel(nn.Module):
    """基于CNN-LSTM的微表情识别模型"""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...] = (20, 256, 256, 4),  # 更新默认输入大小为256x256
                 output_shape: int = 7,
                 cnn_channels: Tuple[int, ...] = (32, 64, 128, 256),  # 增加通道数以处理更大的输入
                 lstm_hidden_size: int = 256,  # 增大隐藏层大小
                 lstm_num_layers: int = 2,
                 dropout_rate: float = 0.5,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 use_batch_norm: bool = True,
                 use_residual: bool = True):
        """初始化CNN-LSTM混合模型
        
        Args:
            input_shape: 输入形状 (sequence_length, height, width, channels)
            output_shape: 输出类别数
            cnn_channels: CNN通道数配置
            lstm_hidden_size: LSTM隐藏层大小
            lstm_num_layers: LSTM层数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向LSTM
            use_attention: 是否使用注意力机制
            use_batch_norm: 是否使用批归一化
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        # 保存参数
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        logger.info(f"初始化CNN-LSTM模型 - 输入形状: {input_shape}, CNN通道: {cnn_channels}")
        
        # CNN特征提取器
        self.cnn_layers = nn.ModuleList()
        in_channels = input_shape[3]  # 输入通道数
        
        for i, out_channels in enumerate(cnn_channels):
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
            
            # 最大池化 - 对于256x256输入，使用更大的池化核心
            if i < len(cnn_channels) - 1:  # 除了最后一层都使用池化
                block.append(nn.MaxPool2d(kernel_size=2))
            
            # Dropout
            block.append(nn.Dropout2d(dropout_rate))
            
            # 添加到模块列表
            self.cnn_layers.append(nn.Sequential(*block))
            
            # 更新输入通道数
            in_channels = out_channels
        
        # 添加自适应池化层，确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 将特征图大小固定为4x4
            
        # 计算CNN输出特征维度
        self.cnn_feature_size = self._get_cnn_feature_size()
        logger.info(f"CNN特征维度: {self.cnn_feature_size}")
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 批归一化层
        if use_batch_norm:
            self.lstm_batch_norm = nn.BatchNorm1d(
                lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
            )
            
        # 注意力层
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden_size * 2 if bidirectional else lstm_hidden_size, 1),
                nn.Tanh()
            )
            
        # 全连接层
        fc_input_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_size // 2, output_shape)
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
        
        # 通过自适应池化层
        x = self.adaptive_pool(x)
            
        # 计算特征维度
        return x.view(1, -1).size(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, height, width, channels)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 检查输入形状
        if x.dim() != 5:
            raise ValueError(f"输入形状必须是5维，但得到了{x.dim()}维")
        
        if seq_len != self.input_shape[0] or x.size(2) != self.input_shape[1] or x.size(3) != self.input_shape[2] or x.size(4) != self.input_shape[3]:
            logger.warning(f"输入形状 {x.shape[1:]} 与期望形状 {self.input_shape} 不匹配")
        
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
        
        # 应用自适应池化
        cnn_features = self.adaptive_pool(cnn_features)
                
        # 重塑特征以便LSTM处理
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(cnn_features)
        
        # 批归一化
        if self.use_batch_norm:
            lstm_out = self.lstm_batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
            
        # 注意力机制
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            lstm_out = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            lstm_out = lstm_out[:, -1]
            
        # 全连接层
        out = self.fc(lstm_out)
        
        return out
        
    def configure_optimizers(self, lr: float = 0.001) -> torch.optim.Optimizer:
        """配置优化器
        
        Args:
            lr: 学习率
            
        Returns:
            优化器实例
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # 添加L2正则化
 
 
 