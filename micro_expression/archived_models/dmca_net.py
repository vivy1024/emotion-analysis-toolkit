"""
DMCA-Net (双流多通道注意力网络) 模型

该模型专为微表情识别设计，针对四通道输入(灰度、光流X、光流Y、LBP)进行优化，
通过通道分组和专门的注意力机制最大化每个通道的贡献。
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChannelAttentionModule(nn.Module):
    """通道注意力模块，用于突出重要通道
    
    使用全局平均池化和最大池化计算通道注意力
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 8):
        """初始化通道注意力模块
        
        Args:
            channels: 输入通道数
            reduction_ratio: 降维比例
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征图，形状为[B, C, T, H, W]
        """
        # 平均池化分支
        avg_out = self.mlp(self.avg_pool(x))
        
        # 最大池化分支
        max_out = self.mlp(self.max_pool(x))
        
        # 融合两个分支
        channel_attention = self.sigmoid(avg_out + max_out)
        
        return x * channel_attention
        
class SpatialAttentionModule(nn.Module):
    """空间注意力模块，用于突出重要的空间区域"""
    
    def __init__(self, kernel_size: int = 7):
        """初始化空间注意力模块
        
        Args:
            kernel_size: 卷积核大小
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), 
                             padding=(0, padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征图，形状为[B, C, T, H, W]
        """
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接平均值和最大值
        attention_input = torch.cat([avg_out, max_out], dim=1)
        
        # 计算空间注意力图
        spatial_attention = self.sigmoid(self.conv(attention_input))
        
        return x * spatial_attention
        
class TemporalAttentionModule(nn.Module):
    """时序注意力模块，用于突出关键时间帧"""
    
    def __init__(self, channels: int):
        """初始化时序注意力模块
        
        Args:
            channels: 输入通道数
        """
        super().__init__()
        self.channels = channels
        # 不在初始化时固定时间序列长度
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征图，形状为[B, C, T, H, W]
        """
        batch_size, channels, t, h, w = x.size()
        
        # 空间池化
        spatial_pool = F.adaptive_avg_pool2d(x.permute(0, 2, 1, 3, 4).reshape(batch_size * t, channels, h, w), 1)
        spatial_pool = spatial_pool.reshape(batch_size, t, channels).permute(0, 2, 1)  # B, C, T
        
        # 动态创建时序注意力层以适应当前时间维度
        temporal_fc = nn.Sequential(
            nn.Linear(t, max(t // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(t // 2, 1), t),
            nn.Sigmoid()
        ).to(x.device)
        
        # 计算时序注意力权重
        temporal_weights = temporal_fc(spatial_pool)  # B, C, T
        
        # 应用权重
        temporal_weights = temporal_weights.unsqueeze(-1).unsqueeze(-1)  # B, C, T, 1, 1
        
        return x * temporal_weights

class ConvBlock(nn.Module):
    """通用卷积块，包含卷积层、批归一化、激活函数和可选的注意力模块"""
    
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: Union[int, Tuple[int, int, int]], 
                stride: Union[int, Tuple[int, int, int]] = 1,
                padding: Union[int, Tuple[int, int, int]] = 0,
                use_attention: bool = False,
                dropout_rate: float = 0.3):
        """初始化卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            use_attention: 是否使用注意力机制
            dropout_rate: Dropout比例
        """
        super().__init__()
        
        # 主要卷积层
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 注意力机制（如果需要）
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttentionModule(out_channels)
            self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            卷积块输出
        """
        # 卷积 + BN + ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # 应用注意力（如果需要）
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        # Dropout
        x = self.dropout(x)
        
        return x

class FeatureFusionModule(nn.Module):
    """特征融合模块，用于融合静态流和动态流的特征"""
    
    def __init__(self, channels: int):
        """初始化特征融合模块
        
        Args:
            channels: 输入通道数
        """
        super().__init__()
        
        self.conv1x1 = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力
        self.channel_attention = ChannelAttentionModule(channels)
        
    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x_static: 静态流特征
            x_dynamic: 动态流特征
            
        Returns:
            融合后的特征
        """
        # 拼接特征
        x_concat = torch.cat([x_static, x_dynamic], dim=1)
        
        # 1x1卷积融合
        x = self.conv1x1(x_concat)
        x = self.bn(x)
        x = self.relu(x)
        
        # 应用通道注意力
        x = self.channel_attention(x)
        
        # 残差连接
        return x + x_static + x_dynamic

class PyramidPoolingModule(nn.Module):
    """金字塔池化模块，替代全连接层"""
    
    def __init__(self, in_channels: int, time_steps: int = 20):
        """初始化金字塔池化模块
        
        Args:
            in_channels: 输入通道数
            time_steps: 时间步数
        """
        super().__init__()
        
        self.pools = nn.ModuleList([
            # 全局池化
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            # 时间维度池化，确保时间维度至少为1
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            # 空间维度池化
            nn.AdaptiveAvgPool3d((1, 2, 2))
        ])
        
        # 计算输出特征数量 (不再包含时间为0的池化)
        self.out_features = in_channels * (1 + 1 + 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            池化后的特征向量
        """
        batch_size = x.size(0)
        features = []
        
        for pool in self.pools:
            out = pool(x)
            out = out.view(batch_size, -1)
            features.append(out)
            
        # 拼接所有池化结果
        return torch.cat(features, dim=1)

class DynamicChannelWeight(nn.Module):
    """动态通道加权模块
    
    自动学习不同通道的重要性权重，可以让模型优先关注重要通道
    使用Gumbel-Softmax实现稀疏权重
    """
    
    def __init__(self, num_channels=4, temp=0.1):
        """初始化动态通道加权模块
        
        Args:
            num_channels: 输入通道数
            temp: 温度系数，控制权重稀疏性
        """
        super().__init__()
        # 可学习的通道权重参数
        self.weights = nn.Parameter(torch.ones(num_channels))
        self.temp = temp
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            加权后的张量
        """
        # Gumbel-Softmax实现稀疏权重
        weighted = F.gumbel_softmax(self.weights, tau=self.temp, hard=False)
        
        # 打印权重（调试用）
        if torch.rand(1).item() < 0.01:  # 1%的概率打印权重
            logger.info(f"通道权重: {weighted.detach().cpu().numpy()}")
            
        return x * weighted.view(1, -1, 1, 1, 1)

class DiagonalMicroAttention(nn.Module):
    """对角微注意力模块
    
    专门针对微表情识别的注意力机制，能够聚焦在细微的面部运动上
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 8):
        """初始化对角微注意力模块
        
        Args:
            channels: 输入通道数
            reduction_ratio: 降维比例
        """
        super().__init__()
        
        # 运动敏感度编码器
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        # 空间细节增强
        self.spatial_enhancer = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        
        # 稀疏激活
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征图
        """
        # 计算时间差分（检测运动）
        motion_map = torch.zeros_like(x)
        motion_map[:, :, 1:] = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        
        # 编码运动特征
        motion_features = self.motion_encoder(motion_map)
        
        # 计算空间注意力图
        # 平均池化和最大池化
        avg_out = torch.mean(motion_features, dim=1, keepdim=True)
        max_out, _ = torch.max(motion_features, dim=1, keepdim=True)
        spatial_attention = self.spatial_enhancer(torch.cat([avg_out, max_out], dim=1))
        
        # 应用注意力
        enhanced_features = x * spatial_attention
        
        return enhanced_features

class MotionSpecificBlock(nn.Module):
    """运动特定卷积块
    
    专为运动特征设计，强调时间维度的卷积
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        """初始化运动特定卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            dropout_rate: Dropout比例
        """
        super().__init__()
        
        # 时间维度卷积（强调运动特征）
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空间维度卷积（较小卷积核）
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            处理后的特征
        """
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return x

class TextureSpecificBlock(nn.Module):
    """纹理特定卷积块
    
    专为静态纹理特征设计，强调空间维度的卷积
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        """初始化纹理特定卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            dropout_rate: Dropout比例
        """
        super().__init__()
        
        # 空间维度卷积（大卷积核，捕捉纹理）
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 时间维度浅层卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            处理后的特征
        """
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x

class FeatureDiscriminator(nn.Module):
    """特征判别器模块
    
    用于对抗训练过程，对输入特征进行真假判别
    有助于消除通道间噪声和冗余信息
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        """初始化特征判别器
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征，形状为[B, feature_dim]
            
        Returns:
            判别结果，形状为[B, 1]，值域为[0, 1]
        """
        return self.discriminator(x)

class DMCANet(nn.Module):
    """双流多通道注意力网络，专为四通道微表情识别设计
    
    将四通道输入分为静态特征流（灰度+LBP）和动态特征流（光流X+Y），
    通过专门的注意力机制和多尺度融合最大化每个通道的贡献。
    
    使用金字塔池化代替全连接层，保留更多时空信息。
    """
    
    def __init__(self, 
                input_shape: Tuple[int, ...] = (20, 256, 256, 4),
                output_shape: int = 7,
                dropout_rate: float = 0.3,
                use_channel_attention: bool = True,
                use_spatial_attention: bool = True,
                use_temporal_attention: bool = True,
                use_dynamic_weights: bool = True,
                use_diagonal_attention: bool = True,
                use_adversarial: bool = False,
                adv_weight: float = 0.1,
                gumbel_temp: float = 0.1,
                device: Union[str, torch.device] = 'cuda'):
        """初始化DMCA-Net模型
        
        Args:
            input_shape: 输入形状，格式为(time_steps, height, width, channels)
            output_shape: 输出类别数
            dropout_rate: Dropout比例
            use_channel_attention: 是否使用通道注意力
            use_spatial_attention: 是否使用空间注意力  
            use_temporal_attention: 是否使用时序注意力
            use_dynamic_weights: 是否使用动态通道加权
            use_diagonal_attention: 是否使用对角微注意力
            use_adversarial: 是否使用对抗训练
            adv_weight: 对抗损失权重
            gumbel_temp: Gumbel-Softmax温度系数
            device: 运行设备
        """
        super().__init__()
        
        # 设置设备
        if isinstance(device, str):
            if device.lower() == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU运行DMCA-Net模型")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = device
            
        logger.info(f"使用设备: {self.device}")
        
        # 保存配置
        self.dropout_rate = dropout_rate
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_dynamic_weights = use_dynamic_weights
        self.use_diagonal_attention = use_diagonal_attention
        self.use_adversarial = use_adversarial
        self.adv_weight = adv_weight
        
        # 解析输入形状
        if len(input_shape) != 4:
            raise ValueError(f"输入形状必须是4维：(time_steps, height, width, channels)，但得到了{len(input_shape)}维")
            
        self.time_steps, self.height, self.width, self.channels = input_shape
        
        if self.channels != 4:
            raise ValueError(f"DMCA-Net需要4通道输入，但得到了{self.channels}通道")
            
        logger.info(f"模型初始化 - 输入形状: time_steps={self.time_steps}, height={self.height}, width={self.width}, channels={self.channels}")
        
        # 动态通道加权（优先级分配）
        if use_dynamic_weights:
            self.channel_weights = DynamicChannelWeight(num_channels=4, temp=gumbel_temp)
            logger.info("使用动态通道加权模块，自动学习通道重要性")
        else:
            self.channel_weights = nn.Identity()
            logger.info("不使用动态通道加权，所有通道权重相等")
            
        # 初始降采样层（共享）- 从256x256降至128x128
        self.initial_downsample = ConvBlock(
            in_channels=4,  # 4通道输入
            out_channels=16,  # 16通道输出
            kernel_size=3,
            stride=2,
            padding=1,
            dropout_rate=dropout_rate
        )
        
        # 静态流（灰度+LBP）- 使用专门化的纹理处理模块
        self.static_stream = nn.Sequential(
            TextureSpecificBlock(2, 16, dropout_rate=dropout_rate),
            TextureSpecificBlock(16, 32, dropout_rate=dropout_rate),
            TextureSpecificBlock(32, 64, dropout_rate=dropout_rate)
        )
        
        # 动态流（光流X+Y）- 使用专门化的运动处理模块
        self.dynamic_stream = nn.Sequential(
            MotionSpecificBlock(2, 16, dropout_rate=dropout_rate),
            MotionSpecificBlock(16, 32, dropout_rate=dropout_rate),
            MotionSpecificBlock(32, 64, dropout_rate=dropout_rate)
        )
        
        # 对角微注意力（专注于动态流上的细微运动）
        if use_diagonal_attention:
            self.diagonal_attention = DiagonalMicroAttention(64)
            logger.info("使用对角微注意力模块，增强微小运动检测")
        
        # 通道注意力（针对各个流）
        if use_channel_attention:
            self.static_channel_attn = ChannelAttentionModule(64)
            self.dynamic_channel_attn = ChannelAttentionModule(64)
            
        # 空间注意力
        if use_spatial_attention:
            self.static_spatial_attn = SpatialAttentionModule()
            self.dynamic_spatial_attn = SpatialAttentionModule()
        
        # 时序注意力（仅用于动态流）
        if use_temporal_attention:
            # 不再需要传递时序维度大小，TemporalAttentionModule会动态适应
            self.temporal_attn = TemporalAttentionModule(64)
        
        # 特征融合模块
        self.fusion = FeatureFusionModule(64)
        
        # 统一特征提取
        self.unified_features = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate)
        )
        
        # 计算经过网络后的特征尺寸
        # 初始：[B, 4, T, H, W] -> [B, 16, T, H/2, W/2]
        # 静态/动态流：[B, 64, T/4, H/16, W/16]
        # 统一特征：[B, 128, T/16, H/64, W/64]
        # 对于初始256x256的输入，最终特征图大小为[B, 128, T/16, 4, 4]
        
        # 金字塔池化模块（替代全连接层）
        self.pyramid_pooling = PyramidPoolingModule(128, self.time_steps // 16)
        
        # 计算金字塔池化输出特征数
        pyramid_features = self.pyramid_pooling.out_features
        
        # 对抗训练组件
        if use_adversarial:
            self.feature_discriminator = FeatureDiscriminator(pyramid_features)
            logger.info("使用对抗训练增强特征表示，减少通道间噪声")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(pyramid_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_shape)
        )
        
        # 调试标志
        self.debug = True
        self._first_forward = True
        
        # 将模型移动到正确的设备
        self.to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"DMCA-Net模型创建完成")
        logger.info(f"总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        logger.info(f"输入形状: {input_shape}")
        logger.info(f"输出类别数: {output_shape}")
        
    def _split_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将输入的四通道数据分为静态流和动态流
        
        Args:
            x: 输入张量，形状为[B, 4, T, H, W]
            
        Returns:
            static_input: 静态流输入，形状为[B, 2, T, H, W]（灰度+LBP）
            dynamic_input: 动态流输入，形状为[B, 2, T, H, W]（光流X+Y）
        """
        # 重新排列通道
        # 通道0: 灰度
        # 通道1: 光流X
        # 通道2: 光流Y
        # 通道3: LBP
        static_input = torch.cat([
            x[:, 0:1],  # 灰度
            x[:, 3:4]   # LBP
        ], dim=1)
        
        dynamic_input = torch.cat([
            x[:, 1:2],  # 光流X
            x[:, 2:3]   # 光流Y
        ], dim=1)
        
        return static_input, dynamic_input
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            return_features: 是否返回中间特征，用于对抗训练
            
        Returns:
            如果return_features为False，返回输出张量，形状为[B, output_shape]
            如果return_features为True，返回(输出张量, 特征张量)元组
        """
        batch_size = x.size(0)
        
        # 调试日志
        if self.debug and self._first_forward:
            logger.info(f"输入形状: {x.shape}")
            
        # 应用动态通道加权
        x = self.channel_weights(x)
            
        # 初始降采样
        x = self.initial_downsample(x)
        
        if self.debug and self._first_forward:
            logger.info(f"初始降采样后形状: {x.shape}")
            
        # 分离通道为静态流和动态流
        static_input, dynamic_input = self._split_channels(x)
        
        if self.debug and self._first_forward:
            logger.info(f"静态流输入形状: {static_input.shape}")
            logger.info(f"动态流输入形状: {dynamic_input.shape}")
            
        # 处理静态流
        static_features = self.static_stream(static_input)
        
        # 处理动态流
        dynamic_features = self.dynamic_stream(dynamic_input)
        
        if self.debug and self._first_forward:
            logger.info(f"静态流特征形状: {static_features.shape}")
            logger.info(f"动态流特征形状: {dynamic_features.shape}")
        
        # 应用对角微注意力（增强微小运动检测）
        if self.use_diagonal_attention:
            dynamic_features = self.diagonal_attention(dynamic_features)
            
        # 应用注意力机制
        if self.use_channel_attention:
            static_features = self.static_channel_attn(static_features)
            dynamic_features = self.dynamic_channel_attn(dynamic_features)
            
        if self.use_spatial_attention:
            static_features = self.static_spatial_attn(static_features)
            dynamic_features = self.dynamic_spatial_attn(dynamic_features)
            
        if self.use_temporal_attention:
            # 时序注意力仅用于动态流（光流）
            dynamic_features = self.temporal_attn(dynamic_features)
            
        # 特征融合
        fused_features = self.fusion(static_features, dynamic_features)
        
        if self.debug and self._first_forward:
            logger.info(f"融合特征形状: {fused_features.shape}")
            
        # 统一特征提取
        features = self.unified_features(fused_features)
        
        if self.debug and self._first_forward:
            logger.info(f"统一特征提取后形状: {features.shape}")
            
        # 金字塔池化
        pooled_features = self.pyramid_pooling(features)
        
        if self.debug and self._first_forward:
            logger.info(f"金字塔池化后形状: {pooled_features.shape}")
            
        # 分类
        outputs = self.classifier(pooled_features)
        
        if self.debug and self._first_forward:
            logger.info(f"输出形状: {outputs.shape}")
            self._first_forward = False
        
        if return_features:
            return outputs, pooled_features
        else:
            return outputs
    
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """调用模型
        
        Args:
            x: 输入张量
            *args: 传递给forward的附加位置参数
            **kwargs: 传递给forward的附加关键字参数
            
        Returns:
            模型输出，具体返回类型取决于forward方法
        """
        return self.forward(x, *args, **kwargs) 