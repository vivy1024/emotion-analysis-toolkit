"""
DMCA-Net增强版 (DMCA-Net++)

本模型针对微表情识别任务进行了优化：
1. 改进的动态通道加权机制，使用Gumbel-Softmax实现硬选择
2. 静态流(灰度+LBP)和动态流(光流)的特征解耦处理
3. 对抗训练组件用于减少噪声和增强泛化能力
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

class ImprovedDynamicChannelWeight(nn.Module):
    """改进的动态通道加权模块
    
    更有效地学习四通道输入(灰度、光流X、光流Y、LBP)的重要性权重
    使用Gumbel-Softmax实现硬选择，提高权重稳定性
    """
    
    def __init__(self, num_channels=4, temp=1.0, hard=True, init_weights=None):
        """初始化改进的动态通道加权模块
        
        Args:
            num_channels: 输入通道数，默认为4
            temp: Gumbel-Softmax温度系数，较高值(>1.0)使权重更平滑
            hard: 是否使用硬选择，强制通道选择更加离散
            init_weights: 初始权重，如果不指定则平均初始化
        """
        super().__init__()
        # 可学习的通道权重参数
        if init_weights is not None:
            # 使用预设的初始权重
            self.weights = nn.Parameter(torch.tensor(init_weights))
        else:
            # 默认平均初始化
            self.weights = nn.Parameter(torch.ones(num_channels))
            
        self.temp = temp
        self.hard = hard
        self.num_channels = num_channels
        
    def forward(self, x: torch.Tensor, au_features: Optional[torch.Tensor] = None, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]或[B, T, H, W, C]
            au_features: AU特征张量，形状为[B, T, au_feature_size]
            return_features: 是否返回中间特征，用于对抗训练
            
        Returns:
            如果return_features为False，返回输出张量，形状为[B, output_shape]
            如果return_features为True，返回(输出张量, 特征张量)元组
        """
        batch_size = x.size(0)
    
    # 添加类属性，缓存输入格式检测结果
    if not hasattr(self, '_input_format_detected'):
        self._input_format_detected = False
        self._needs_permute = False
        
    # 只在第一次前向传播时检测输入格式
    if not self._input_format_detected:
        # 检测输入格式是否为[B, T, H, W, C]
        if (len(x.shape) == 5 and 
            x.shape[1] == self.time_steps and
            x.shape[4] == self.channels):
            logger.info(f"首次检测: 输入格式为[B, T, H, W, C]，将进行转换为[B, C, T, H, W]")
            self._needs_permute = True
        else:
            logger.info(f"首次检测: 输入格式已为[B, C, T, H, W]，无需转换")
            self._needs_permute = False
            
        self._input_format_detected = True
    
    # 根据缓存的检测结果决定是否需要转换
    if self._needs_permute:
        x = x.permute(0, 4, 1, 2, 3)
    
    # 应用动态通道加权
    if self.use_dynamic_weights:
        x = self.channel_weights(x)

class TextureSpecificBlock(nn.Module):
    """纹理特定卷积块 - 增强版
    
    专为静态纹理特征(灰度+LBP)设计
    以空间维度的卷积为主，较小的时间维度卷积
    增加了注意力机制
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        """初始化纹理特定卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            dropout_rate: Dropout比例
        """
        super().__init__()
        
        # 首先进行空间维度卷积（大卷积核，捕捉纹理）
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 非局部注意力块，增强长程依赖
        self.non_local = NonLocalBlock(out_channels)
        
        # 轻量时间维度卷积
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
        x = self.non_local(x)
        x = self.temporal_conv(x)
        return x

class MotionSpecificBlock(nn.Module):
    """运动特定卷积块 - 增强版
    
    专为动态运动特征(光流X+Y)设计
    以时间维度的卷积为主，更强调运动特征
    添加了时间移位模块
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
        
        # 时间移位模块，增强时序特征
        self.tsm = TemporalShiftModule(out_channels)
        
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
        x = self.tsm(x)
        x = self.spatial_conv(x)
        return x

class TemporalShiftModule(nn.Module):
    """时间移位模块
    
    通过在通道维度上沿时间轴移位，增强时序建模能力
    不增加参数量和计算复杂度
    """
    
    def __init__(self, channels: int, shift_ratio: float = 0.25):
        """初始化时间移位模块
        
        Args:
            channels: 输入通道数
            shift_ratio: 需要移位的通道比例
        """
        super().__init__()
        self.channels = channels
        self.shift_ratio = shift_ratio
        self.shift_channels = int(channels * shift_ratio)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            时序增强的特征
        """
        # 获取张量尺寸
        B, C, T, H, W = x.size()
        
        # 原始张量副本
        out = x.clone()
        
        # 左移通道
        out[:, :self.shift_channels, 1:, :, :] = x[:, :self.shift_channels, :-1, :, :]
        
        # 右移通道
        out[:, self.shift_channels:2*self.shift_channels, :-1, :, :] = x[:, self.shift_channels:2*self.shift_channels, 1:, :, :]
        
        return out

class NonLocalBlock(nn.Module):
    """非局部注意力块
    
    捕获长距离空间依赖关系，增强纹理表示
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 2, mode: str = "memory_efficient"):
        """初始化非局部块
        
        Args:
            in_channels: 输入通道数
            reduction_ratio: 中间层通道减少比例
            mode: 计算模式，可选"normal"或"memory_efficient"
        """
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction_ratio
        self.mode = mode
        
        # 键值查询投影
        self.query = nn.Conv3d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        # 缩放因子
        self.scale = self.reduced_channels ** -0.5
        
        # 内存高效模式的空间降采样
        if mode == "memory_efficient":
            # 空间降采样
            self.downsample = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征
        """
        batch_size = x.size(0)
        
        # 使用内存高效模式
        if self.mode == "memory_efficient":
            return self._forward_memory_efficient(x)
        
        # 标准模式 - 可能占用大量内存
        # 计算查询、键、值
        q = self.query(x).view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)  # B, THW, C'
        k = self.key(x).view(batch_size, self.reduced_channels, -1)  # B, C', THW
        v = self.value(x).view(batch_size, self.in_channels, -1).permute(0, 2, 1)  # B, THW, C
        
        # 计算注意力
        attn = torch.bmm(q, k) * self.scale  # B, THW, THW
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attn, v)  # B, THW, C
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.in_channels, *x.size()[2:])
        
        # 残差连接
        out = self.output_conv(out) + x
        
        return out
    
    def _forward_memory_efficient(self, x: torch.Tensor) -> torch.Tensor:
        """内存高效的前向传播
        
        通过空间降采样和分块处理减少内存使用
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]
            
        Returns:
            注意力加权的特征
        """
        batch_size, channels, time_steps, height, width = x.size()
        
        # 空间降采样以减少计算量和内存使用
        x_down = self.downsample(x)
        
        # 计算查询、键、值 (使用降采样后的特征)
        q = self.query(x_down)
        k = self.key(x_down)
        v = self.value(x_down)
        
        # 压缩空间维度
        q_flat = q.view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)  # B, T(H/2)(W/2), C'
        k_flat = k.view(batch_size, self.reduced_channels, -1)  # B, C', T(H/2)(W/2)
        v_flat = v.view(batch_size, channels, -1).permute(0, 2, 1)  # B, T(H/2)(W/2), C
        
        # 计算注意力得分 - 关键的内存密集操作
        attn = torch.bmm(q_flat, k_flat) * self.scale  # B, T(H/2)(W/2), T(H/2)(W/2)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out_flat = torch.bmm(attn, v_flat)  # B, T(H/2)(W/2), C
        out = out_flat.permute(0, 2, 1).view(batch_size, channels, *x_down.size()[2:])
        
        # 上采样回原始分辨率
        out = self.upsample(out)
        
        # 残差连接
        out = self.output_conv(out) + x
        
        return out

class ImprovedFeatureFusionModule(nn.Module):
    """改进的特征融合模块
    
    使用注意力机制和自适应权重融合静态流和动态流特征
    """
    
    def __init__(self, channels: int):
        """初始化改进的特征融合模块
        
        Args:
            channels: 每个流的输入通道数
        """
        super().__init__()
        
        # 静态流通道调整
        self.static_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        # 动态流通道调整
        self.dynamic_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        # 融合注意力
        self.fusion_attn = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 2, kernel_size=1),  # 2个权重，一个给静态流，一个给动态流
            nn.Softmax(dim=1)
        )
        
        # 融合后的处理
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_static: torch.Tensor, x_dynamic: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x_static: 静态流特征
            x_dynamic: 动态流特征
            
        Returns:
            融合后的特征
        """
        # 调整每个流的特征
        static_feat = self.static_conv(x_static)
        dynamic_feat = self.dynamic_conv(x_dynamic)
        
        # 计算融合注意力
        concat_feat = torch.cat([static_feat, dynamic_feat], dim=1)
        fusion_weights = self.fusion_attn(concat_feat)  # B, 2, T, H, W
        
        # 应用权重
        weighted_static = static_feat * fusion_weights[:, 0:1]
        weighted_dynamic = dynamic_feat * fusion_weights[:, 1:2]
        
        # 加权求和
        fused = weighted_static + weighted_dynamic
        
        # 融合后处理
        out = self.fusion_conv(fused)
        
        # 残差连接
        out = out + static_feat + dynamic_feat
        
        return out

class FeatureDiscriminator(nn.Module):
    """特征判别器
    
    用于对抗训练，增强模型泛化能力
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        """初始化特征判别器
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            判别结果 (0-1之间的值)
        """
        return self.discriminator(x)

class PyramidPoolingModule(nn.Module):
    """金字塔池化模块
    
    多尺度特征提取，替代全连接层
    """
    
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
            # 时间维度池化
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            # 空间维度池化
            nn.AdaptiveAvgPool3d((1, 2, 2))
        ])
        
        # 计算输出特征数量
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

class DMCANetEnhanced(nn.Module):
    """增强版DMCA-Net (DMCA-Net++)
    
    针对四通道输入优化的微表情识别模型
    主要增强点:
    1. 改进的动态通道加权
    2. 静态/动态特征解耦处理
    3. 对抗训练支持
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, ...], 
        output_shape: int, 
        dropout_rate: float = 0.5,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        use_dynamic_weights: bool = True,
        use_diagonal_attention: bool = True,
        use_adversarial: bool = False,
        use_au_features: bool = False,
        au_feature_size: int = 35,
        au_fusion_method: str = 'concat',
        device: Union[str, torch.device] = "cpu"
    ):
        """初始化DMCA-Net增强模型
        
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
            use_au_features: 是否使用AU特征
            au_feature_size: AU特征大小
            au_fusion_method: AU特征融合方法，可选'concat'、'add'或'gate'
            device: 运行设备
        """
        super().__init__()
        
        # 保存参数
        self.time_steps, self.height, self.width, self.channels = input_shape
        self.input_shape = input_shape  # 保存完整的输入形状
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_dynamic_weights = use_dynamic_weights
        self.use_diagonal_attention = use_diagonal_attention
        self.use_adversarial = use_adversarial
        
        # AU特征相关参数
        self.use_au_features = use_au_features
        self.au_feature_size = au_feature_size
        self.au_fusion_method = au_fusion_method
        
        # 确保通道数合理，至少为4，或与输入通道数匹配
        self.num_channels = max(4, self.channels)
        
        # 动态通道加权  - 修改通道数为实际输入通道数
        if self.use_dynamic_weights:
            self.channel_weights = ImprovedDynamicChannelWeight(self.channels, temp=1.0, hard=False)
            
        # 修复初始化下采样层，处理张量维度顺序
        self.permute_dims = False  # 始终设置为False，避免不必要的维度转换
        # 如果需要更改回原先的自动检测逻辑，取消下面注释
        # if self.time_steps >= 10:  # 如果时间步数大于通道数，可能维度顺序不同
        #     self.permute_dims = True
        #     logger.info(f"检测到输入维度可能是[B, T, H, W, C]格式，将进行维度转换")
        
        # 第一层卷积，降采样
        self.initial_downsample = nn.Sequential(
            nn.Conv3d(
        # 始终使用self.channels作为输入通道数
                self.channels,  # 修改这一行，移除条件判断
                16, 
                kernel_size=(1, 3, 3), 
                stride=(1, 2, 2), 
                padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
        
        # 静态特征流处理 - 修改为更灵活的通道配置
        static_in_channels = max(1, self.channels // 2 + (self.channels % 2))
        self.static_stream = nn.Sequential(
            TextureSpecificBlock(16, 32, dropout_rate=0.3),  # 修改为接受16通道输入
            TextureSpecificBlock(32, 48, dropout_rate=0.4),
            TextureSpecificBlock(48, 64, dropout_rate=self.dropout_rate)
        )
        
        # 动态特征流处理 - 修改为更灵活的通道配置
        dynamic_in_channels = max(1, self.channels // 2)
        self.dynamic_stream = nn.Sequential(
            MotionSpecificBlock(16, 32, dropout_rate=0.3),  # 修改为接受16通道输入
            MotionSpecificBlock(32, 48, dropout_rate=0.4),
            MotionSpecificBlock(48, 64, dropout_rate=self.dropout_rate)
        )
        
        # 注意力机制
        if self.use_channel_attention:
            self.static_channel_attn = self._create_channel_attention(64)
            self.dynamic_channel_attn = self._create_channel_attention(64)
            
        if self.use_spatial_attention:
            self.static_spatial_attn = self._create_spatial_attention()
            self.dynamic_spatial_attn = self._create_spatial_attention()
            
        if self.use_temporal_attention:
            self.temporal_attn = self._create_temporal_attention(64)
        
        # 对角微注意力
        if self.use_diagonal_attention:
            self.diagonal_attention = self._create_diagonal_attention(64)
            
        # 双流特征融合
        self.fusion = ImprovedFeatureFusionModule(64)
        
        # 特征提取块
        self.unified_features = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate)
        )
        
        # AU特征处理
        if self.use_au_features:
            # 使用RNN处理AU特征序列
            self.au_lstm = nn.LSTM(
                input_size=self.au_feature_size,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # AU特征注意力
            self.au_attention = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            
            # AU特征投影
            self.au_projection = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            
            # 融合门控
            if self.au_fusion_method == 'gate':
                self.fusion_gate = nn.Sequential(
                    nn.Linear(128 + 128, 128),
                    nn.Sigmoid()
                )
            
            # 全局平均池化
            self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
            
            # 分类器，根据不同的融合方法调整输入维度
            if self.au_fusion_method == 'concat':
                # 空间特征和AU特征拼接
                classifier_input = 128 * 6 + 128  # 5层金字塔池化+AU特征
            else:
                # 使用add或gate时，只使用融合后的特征
                classifier_input = 128
        else:
            # 无AU特征，使用标准金字塔池化特征
            classifier_input = 128 * 6  # 5层金字塔池化
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.output_shape)
        )
        # 初始化金字塔池化模块
        self.pyramid_pooling = PyramidPoolingModule(128, time_steps=self.time_steps)
        # 对抗性判别器（如果启用）
        if self.use_adversarial:
            self.domain_classifier = FeatureDiscriminator(classifier_input)
            
        # 打印和记录模型配置
        self.log_model_config()
        
        # 将模型移动到正确的设备
        self.to(device)
    
    def _create_channel_attention(self, channels: int, reduction_ratio: int = 8):
        """创建通道注意力模块
        
        Args:
            channels: 输入通道数
            reduction_ratio: 降维比例
            
        Returns:
            通道注意力模块
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def _create_spatial_attention(self, kernel_size: int = 7):
        """创建空间注意力模块
        
        Args:
            kernel_size: 卷积核大小
            
        Returns:
            空间注意力模块
        """
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), 
                    padding=(0, padding, padding), bias=False),
            nn.Sigmoid()
        )
    
    def _create_temporal_attention(self, channels: int):
        """创建时序注意力模块
        
        Args:
            channels: 输入通道数
            
        Returns:
            时序注意力模块
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 空间池化
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _create_diagonal_attention(self, channels: int, reduction_ratio: int = 8):
        """创建对角微注意力模块
        
        Args:
            channels: 输入通道数
            reduction_ratio: 降维比例
            
        Returns:
            对角微注意力模块
        """
        return nn.Sequential(
            # 运动敏感度编码器
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            # 空间细节增强
            nn.Sigmoid()
        )
    
    def log_model_config(self):
        """打印和记录模型配置"""
        logger.info(f"DMCA-Net++模型创建完成")
        logger.info(f"输入形状: {self.input_shape}")
        logger.info(f"输出类别数: {self.output_shape}")
        logger.info(f"Dropout比例: {self.dropout_rate}")
        logger.info(f"使用通道注意力: {self.use_channel_attention}")
        logger.info(f"使用空间注意力: {self.use_spatial_attention}")
        logger.info(f"使用时序注意力: {self.use_temporal_attention}")
        logger.info(f"使用动态通道加权: {self.use_dynamic_weights}")
        logger.info(f"使用对角微注意力: {self.use_diagonal_attention}")
        logger.info(f"使用对抗训练: {self.use_adversarial}")
        logger.info(f"使用AU特征: {self.use_au_features}")
        logger.info(f"AU特征大小: {self.au_feature_size}")
        logger.info(f"AU特征融合方法: {self.au_fusion_method}")
    
    def forward(self, x: torch.Tensor, au_features: Optional[torch.Tensor] = None, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量，形状为[B, C, T, H, W]或[B, T, H, W, C]
            au_features: AU特征张量，形状为[B, T, au_feature_size]
            return_features: 是否返回中间特征，用于对抗训练
            
        Returns:
            如果return_features为False，返回输出张量，形状为[B, output_shape]
            如果return_features为True，返回(输出张量, 特征张量)元组
        """
        batch_size = x.size(0)
        
        # 自动检测并调整输入维度顺序，不依赖self.permute_dims
        # 如果输入形状是[B, T, H, W, C]，转换为[B, C, T, H, W]
        if (len(x.shape) == 5 and 
            x.shape[1] == self.time_steps and
            x.shape[4] == self.channels):
            logger.info(f"检测到输入格式为[B, T, H, W, C]，转换为[B, C, T, H, W]")
            x = x.permute(0, 4, 1, 2, 3)
            
        # 应用动态通道加权
        if self.use_dynamic_weights:
            x = self.channel_weights(x)
            
        # 输出初始输入形状以供调试
        logger.debug(f"卷积前输入形状: {x.shape}")
        
        # 初始降采样
        x = self.initial_downsample(x)
        
        # 输出降采样后形状以供调试
        logger.debug(f"卷积后形状: {x.shape}")
        
        # 处理静态流
        static_features = self.static_stream(x)
        
        # 处理动态流
        dynamic_features = self.dynamic_stream(x)
        
        # 应用注意力机制
        if self.use_channel_attention:
            static_features = static_features * self.static_channel_attn(static_features)
            dynamic_features = dynamic_features * self.dynamic_channel_attn(dynamic_features)
            
        if self.use_spatial_attention:
            # 计算空间注意力图
            static_spatial = torch.cat([
                torch.mean(static_features, dim=1, keepdim=True),
                torch.max(static_features, dim=1, keepdim=True)[0]
            ], dim=1)
            
            dynamic_spatial = torch.cat([
                torch.mean(dynamic_features, dim=1, keepdim=True),
                torch.max(dynamic_features, dim=1, keepdim=True)[0]
            ], dim=1)
            
            static_features = static_features * self.static_spatial_attn(static_spatial)
            dynamic_features = dynamic_features * self.dynamic_spatial_attn(dynamic_spatial)
        
        if self.use_temporal_attention:
            try:
                # 时序注意力仅用于动态流（光流）
                # 空间压缩
                b, c, t, h, w = dynamic_features.size()
                spatial_pool = dynamic_features.view(b, c, t, -1).mean(-1)  # B, C, T
                temporal_attn = self.temporal_attn(spatial_pool).unsqueeze(-1).unsqueeze(-1)  # B, C, T, 1, 1
                dynamic_features = dynamic_features * temporal_attn
            except Exception as e:
                # 出错时记录并继续，不影响整体训练
                logger.warning(f"应用时序注意力时出错: {str(e)}")
                logger.warning(f"跳过时序注意力，继续处理")
            
        # 特征融合
        fused_features = self.fusion(static_features, dynamic_features)
        
        # 统一特征提取
        features = self.unified_features(fused_features)
        
        # 处理AU特征
        au_output = None
        if self.use_au_features and au_features is not None:
            try:
                # AU特征通过LSTM处理
                au_out, _ = self.au_lstm(au_features)  # [B, T, au_lstm_hidden_size*2]
                
                # 应用注意力获取时序汇总特征
                au_attention_weights = self.au_attention(au_out)  # [B, T, 1]
                au_attention_weights = F.softmax(au_attention_weights, dim=1)
                au_output = torch.sum(au_attention_weights * au_out, dim=1)  # [B, au_lstm_hidden_size*2]
                
                logger.debug(f"处理AU特征: 形状={au_features.shape}, 输出形状={au_output.shape}")
            except Exception as e:
                logger.warning(f"处理AU特征时出错: {e}")
                self.use_au_features = False  # 降级处理
        
        # 金字塔池化前的特征融合
        if self.use_au_features and au_output is not None:
            batch_size = features.size(0)
            # 先对空间特征进行全局池化以便融合
            pooled_spatial = self.global_avg_pool(features)
            pooled_spatial = pooled_spatial.view(batch_size, -1)  # [B, 128]
            
            # 根据融合方法处理
            if self.au_fusion_method == 'concat':
                # 简单拼接，在金字塔池化后手动拼接
                pyramid_features = self.pyramid_pooling(features)
                combined_features = torch.cat([pyramid_features, au_output], dim=1)
                
            elif self.au_fusion_method == 'add':
                # 投影AU特征，然后相加
                projected_au = self.au_projection(au_output)  # [B, 128]
                combined_spatial = pooled_spatial + projected_au
                # 重塑以便金字塔池化 - 这里只用全局池化特征
                combined_features = combined_spatial
                
            elif self.au_fusion_method == 'gate':
                # 门控融合
                concat_features = torch.cat([pooled_spatial, au_output], dim=1)
                gate = self.fusion_gate(concat_features)
                combined_spatial = pooled_spatial * gate + self.au_projection(au_output) * (1 - gate)
                # 同样只使用融合后的全局池化特征
                combined_features = combined_spatial
        else:
            # 无AU特征，正常进行金字塔池化
            combined_features = self.pyramid_pooling(features)
        
        # 分类
        if self.use_au_features and au_output is not None and self.au_fusion_method == 'concat':
            # 已经在上面拼接过特征
            output = self.classifier(combined_features)
        else:
            # 无AU特征或非拼接方法，使用常规处理
            output = self.classifier(combined_features)
        
        if return_features:
            return output, combined_features
        else:
            return output 

    @classmethod
    def create_for_device(cls, input_shape: Tuple[int, ...], output_shape: int, 
                         use_au_features: bool = False,
                         au_feature_size: int = 35,
                         au_fusion_method: str = 'concat',
                         device: Union[str, torch.device] = "cuda") -> "DMCANetEnhanced":
        """根据设备创建适合的模型配置
        
        根据可用GPU显存自动调整模型复杂度
        
        Args:
            input_shape: 输入形状，格式为(time_steps, height, width, channels)
            output_shape: 输出类别数
            use_au_features: 是否使用AU特征
            au_feature_size: AU特征维度
            au_fusion_method: AU特征融合方法
            device: 运行设备
            
        Returns:
            配置优化的DMCANetEnhanced模型实例
        """
        # 检测显存
        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device
            
        vram_config = "HIGH"  # 默认高配置
        
        if device_obj.type == "cuda":
            try:
                gpu_idx = device_obj.index if device_obj.index is not None else 0
                # 获取显存信息（GB）
                free_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                
                # 根据显存大小选择配置
                if free_memory < 6:  # 修改：将阈值从8GB改为6GB
                    vram_config = "LOW"
                    logger.info(f"低端显卡 ({free_memory:.1f}GB)，使用最小化模型配置")
                elif free_memory < 12:
                    vram_config = "MEDIUM"
                    logger.info(f"中端显卡 ({free_memory:.1f}GB)，使用中等模型配置")
                else:
                    logger.info(f"高端显卡 ({free_memory:.1f}GB)，使用完整模型功能")
            except Exception as e:
                logger.warning(f"无法确定显存大小，将使用中等配置: {str(e)}")
                vram_config = "MEDIUM"
        else:
            logger.info("使用CPU运行，选择较小模型配置")
            vram_config = "LOW"
        
        # 根据配置选择合适的参数
        if vram_config == "LOW":
            # 低配置：禁用大部分高级功能，降低模型复杂度
            return cls(
                input_shape=input_shape,
                output_shape=output_shape,
                dropout_rate=0.5,  # 增加dropout防止过拟合
                use_channel_attention=True,  # 保留通道注意力（核心功能）
                use_spatial_attention=False,  # 禁用空间注意力
                use_temporal_attention=False,  # 明确禁用时序注意力
                use_diagonal_attention=False,  # 禁用对角微注意力
                use_dynamic_weights=True,     # 保留动态通道加权
                use_au_features=use_au_features,  # 使用传入的参数
                au_feature_size=au_feature_size,
                au_fusion_method=au_fusion_method,
                device=device
            )
        elif vram_config == "MEDIUM":
            # 中配置：保留部分高级功能
            return cls(
                input_shape=input_shape,
                output_shape=output_shape,
                dropout_rate=0.4,
                use_channel_attention=True,
                use_spatial_attention=True,
                use_temporal_attention=False,  # 明确禁用时序注意力
                use_diagonal_attention=True,
                use_dynamic_weights=True,
                use_au_features=use_au_features,  # 使用传入的参数
                au_feature_size=au_feature_size,
                au_fusion_method=au_fusion_method,
                device=device
            )
        else:  # HIGH
            # 高配置：启用全部功能
            return cls(
                input_shape=input_shape,
                output_shape=output_shape,
                dropout_rate=0.3,
                use_channel_attention=True,
                use_spatial_attention=True,
                use_temporal_attention=True,  # 高端配置启用时序注意力
                use_diagonal_attention=True,
                use_dynamic_weights=True,
                use_au_features=use_au_features,  # 使用传入的参数
                au_feature_size=au_feature_size,
                au_fusion_method=au_fusion_method,
                device=device
            ) 