#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观表情识别模型定义
包含用于表情识别的各种神经网络模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

class BasicConvBlock(nn.Module):
    """基本卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class MacroExpressionModel(nn.Module):
    """
    宏观表情识别模型
    基于ResNet架构的表情识别模型，针对FER2013等数据集优化
    """
    
    def __init__(self, num_classes: int = 7, grayscale: bool = False):
        """
        初始化宏观表情识别模型
        
        Args:
            num_classes: 输出类别数
            grayscale: 是否使用灰度图像输入
        """
        super(MacroExpressionModel, self).__init__()
        
        # 输入通道数（1或3）
        self.in_channels = 1 if grayscale else 3
        self.num_classes = num_classes
        
        # 初始卷积层
        self.conv1 = BasicConvBlock(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第一个残差块
        self.residual1_1 = self._make_residual_block(64, 64)
        self.residual1_2 = self._make_residual_block(64, 64)
        
        # 缩减特征图尺寸
        self.conv3 = BasicConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        
        # 第二个残差块
        self.residual2_1 = self._make_residual_block(128, 128)
        self.residual2_2 = self._make_residual_block(128, 128)
        
        # 缩减特征图尺寸
        self.conv4 = BasicConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        
        # 空间注意力机制
        self.spatial_attention = self._make_attention_block(256)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """创建残差块"""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def _make_attention_block(self, channels: int) -> nn.Module:
        """创建空间注意力块"""
        return nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第一个残差块
        identity = x
        out = self.residual1_1(x)
        out += identity
        
        identity = out
        out = self.residual1_2(out)
        out += identity
        
        # 缩减特征图尺寸
        x = self.conv3(out)
        
        # 第二个残差块
        identity = x
        out = self.residual2_1(x)
        out += identity
        
        identity = out
        out = self.residual2_2(out)
        out += identity
        
        # 缩减特征图尺寸
        x = self.conv4(out)
        
        # 应用空间注意力
        attention_map = self.spatial_attention(x)
        x = x * attention_map
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class LightweightEmotionModel(nn.Module):
    """
    轻量级表情识别模型
    针对移动设备和低功耗设备优化的轻量级模型
    """
    
    def __init__(self, num_classes: int = 7, grayscale: bool = False):
        """
        初始化轻量级表情识别模型
        
        Args:
            num_classes: 输出类别数
            grayscale: 是否使用灰度图像输入
        """
        super(LightweightEmotionModel, self).__init__()
        
        # 输入通道数（1或3）
        self.in_channels = 1 if grayscale else 3
        self.num_classes = num_classes
        
        # 使用深度可分离卷积减少参数量
        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 深度可分离卷积1
        self.depthwise1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16)
        self.pointwise1 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 深度可分离卷积2
        self.depthwise2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 深度可分离卷积1
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 深度可分离卷积2
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 