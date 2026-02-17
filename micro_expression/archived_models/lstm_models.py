#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM模型模块 - 提供LSTM模型实现

该模块提供了基于LSTM的微表情识别模型实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroExpressionLSTMModel(nn.Module):
    """基于LSTM的微表情识别模型"""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...] = (20, 256, 256, 4),  # 更新默认输入大小为256x256
                 output_shape: int = 7,
                 hidden_size: int = 256,  # 增大隐藏层大小以处理更多特征
                 num_layers: int = 2,
                 dropout_rate: float = 0.5,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 use_batch_norm: bool = True):
        """初始化LSTM模型
        
        Args:
            input_shape: 输入形状 (sequence_length, height, width, channels)
            output_shape: 输出类别数
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向LSTM
            use_attention: 是否使用注意力机制
            use_batch_norm: 是否使用批归一化
        """
        super().__init__()
        
        # 保存参数
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        
        # 计算输入特征维度
        sequence_length, height, width, channels = input_shape
        self.feature_size = height * width * channels
        
        logger.info(f"初始化LSTM模型 - 输入形状: {input_shape}, 特征大小: {self.feature_size}")
        
        # 由于256x256输入特征维度过大，先通过线性层降维
        self.feature_reduction = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024)
        )
        reduced_feature_size = 1024
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=reduced_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 批归一化层
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(
                hidden_size * 2 if bidirectional else hidden_size
            )
            
        # 注意力层
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1),
                nn.Tanh()
            )
            
        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_size // 2, output_shape)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, height, width, channels)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 检查输入形状
        if x.dim() != 5:
            raise ValueError(f"输入形状必须是5维，但得到了{x.dim()}维")
        
        if x.size(1) != self.input_shape[0] or x.size(2) != self.input_shape[1] or x.size(3) != self.input_shape[2] or x.size(4) != self.input_shape[3]:
            logger.warning(f"输入形状 {x.shape[1:]} 与期望形状 {self.input_shape} 不匹配")
        
        # 重塑输入
        x = x.view(batch_size, self.input_shape[0], -1)  # (batch_size, sequence_length, features)
        
        # 特征降维
        sequence_length = x.size(1)
        x_reshaped = x.view(batch_size * sequence_length, -1)
        x_reduced = self.feature_reduction(x_reshaped)
        x = x_reduced.view(batch_size, sequence_length, -1)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size * num_directions)
        
        # 批归一化
        if self.use_batch_norm:
            lstm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
            
        # 注意力机制
        if self.use_attention:
            attention_weights = self.attention(lstm_out)  # (batch_size, sequence_length, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            lstm_out = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size * num_directions)
        else:
            lstm_out = lstm_out[:, -1]  # 取最后一个时间步的输出
            
        # 全连接层
        out = self.fc(lstm_out)  # (batch_size, num_classes)
        
        return out
        
    def configure_optimizers(self, lr: float = 0.001) -> torch.optim.Optimizer:
        """配置优化器
        
        Args:
            lr: 学习率
            
        Returns:
            优化器实例
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # 添加L2正则化
 
 
 