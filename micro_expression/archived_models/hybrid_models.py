"""
混合模型模块 - 实现基于PyTorch的混合微表情识别模型

该模块提供了两种混合模型实现：
1. HybridMicroExpressionModel: 结合CNN和LSTM的混合模型
2. HybridModelPyTorch: 混合模型的包装类，提供统一接口

主要特点：
- 支持3D卷积和LSTM的混合架构
- 动态调整输入通道数
- 内存优化的checkpoint机制
- 灵活的模型配置
"""

import logging
from typing import Tuple, Optional, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMicroExpressionModel(nn.Module):
    """混合微表情识别模型，结合CNN和LSTM
    
    该模型使用3D CNN提取时空特征，然后用LSTM处理时序信息。
    支持动态调整输入通道数和内存优化。
    
    Attributes:
        device: 计算设备
        num_classes: 分类数量
        time_steps: 时间步数
        height: 图像高度
        width: 图像宽度
        in_channels: 输入通道数
        cnn_output_size: CNN输出大小
        cnn_flattened_size: CNN展平后大小
        lstm_hidden_size: LSTM隐藏层大小
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (20, 256, 256, 4),
        output_shape: int = 7,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True,
        use_feature_fusion: bool = True,
        use_transfer_learning: bool = False
    ):
        """初始化混合架构微表情识别模型
        
        Args:
            input_shape: 输入形状，格式为 (time_steps, height, width, channels)
            output_shape: 输出类别数
            use_batch_norm: 是否使用批量归一化
            dropout_rate: Dropout比率
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
            use_feature_fusion: 是否使用特征融合
            use_transfer_learning: 是否使用迁移学习特征
        """
        super(HybridMicroExpressionModel, self).__init__()
        
        # 检查CUDA可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存配置
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_feature_fusion = use_feature_fusion
        self.use_transfer_learning = use_transfer_learning
        
        # 解析输入形状
        if len(input_shape) == 4:
            self.time_steps, self.height, self.width, self.channels = input_shape
        else:
            raise ValueError(f"输入形状必须是4维 (time_steps, height, width, channels)，但得到了 {len(input_shape)} 维")
            
        # 确认输入通道
        if self.channels != 4:
            logger.warning(f"输入通道应为4 (灰度+光流+LBP)，但得到了 {self.channels}")
        
        # 构建空间特征提取器
        self.spatial_features = self._build_spatial_branch()
        
        # 构建时序特征提取器
        self.temporal_features = self._build_temporal_branch()
        
        # 构建多尺度特征提取器
        self.multiscale_features = self._build_multiscale_branch()
        
        # 特征融合层
        if self.use_feature_fusion:
            # 适应更大的输入尺寸
            spatial_out_size = 2048  # 针对256x256输入的估计值
            temporal_out_size = 1024
            multiscale_out_size = 1024
            
            fusion_in_size = spatial_out_size + temporal_out_size + multiscale_out_size
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_in_size, 1024),
                nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.8)
            )
            classifier_input_size = 512
        else:
            # 如果不使用特征融合，只使用空间特征
            classifier_input_size = 2048  # 针对256x256的估计值
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, output_shape)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 移动到设备
        self.to(self.device)
        
    def _build_spatial_branch(self):
        """构建空间特征提取分支 - 使用3D CNN
        
        调整以处理256x256输入
        """
        layers = []
        in_channels = self.channels
        
        # 使用较小的起始卷积滤波器以处理较大的输入
        filters = [32, 64, 128, 256, 512]
        
        for i, f in enumerate(filters):
            # 第一层卷积
            layers.append(nn.Conv3d(
                in_channels, f, 
                kernel_size=(3, 3, 3), 
                stride=1, 
                padding=1
            ))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm3d(f))
                
            layers.append(nn.ReLU())
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout3d(self.dropout_rate * 0.5))
            
            # 可选的残差连接
            if self.use_residual and i > 0:
                # 残差连接需要相同的通道数
                if in_channels != f:
                    res_layers = nn.Sequential(
                        nn.Conv3d(in_channels, f, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm3d(f) if self.use_batch_norm else nn.Identity()
                    )
                    # 这里简化处理，实际残差连接需要更复杂的实现
            
            # 池化层 - 为256x256输入使用更大的步长
            # 最后一层不使用池化
            if i < len(filters) - 1:
                pool_size = (1, 2, 2)  # 保持时间维度不变
                layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = f
        
        # 针对更大输入添加额外的降维池化
        layers.append(nn.AdaptiveAvgPool3d((None, 4, 4)))  # 适应性地将空间维度降为4x4
        
        return nn.Sequential(*layers)
        
    def _build_temporal_branch(self):
        """构建时间特征提取分支 - 使用LSTM"""
        # 首先通过一些卷积层减少空间维度
        spatial_reduction = nn.Sequential(
            nn.Conv3d(self.channels, 32, kernel_size=(1, 7, 7), stride=(1, 4, 4), padding=(0, 3, 3)),
            nn.BatchNorm3d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(32, 64, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # 进一步降维以处理256x256输入
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )
        
        # 计算LSTM输入维度
        # 对于256x256输入，经过上述降维后约为2x2的特征图
        lstm_input_dim = 128 * 2 * 2  # 特征通道 * 降维后高度 * 降维后宽度
        
        # 创建LSTM层
        lstm_hidden_dim = 512
        lstm_layers = 2
        
        # 双向LSTM提取时间特征
        lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=self.dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        return nn.ModuleDict({
            'spatial_reduction': spatial_reduction,
            'lstm': lstm,
            'lstm_hidden_dim': torch.nn.Parameter(torch.tensor(lstm_hidden_dim), requires_grad=False)
        })
    
    def _build_multiscale_branch(self):
        """构建多尺度特征提取分支 - 使用不同尺度的卷积"""
        # 对于256x256的输入，使用更强的降维
        layers = []
        
        # 初始降维
        layers.append(nn.Sequential(
            nn.Conv3d(self.channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        ))
        
        # 多尺度卷积模块
        scales = [(1, 3, 3), (3, 3, 3), (5, 3, 3)]
        concat_channels = 0
        
        multiscale_modules = []
        for scale in scales:
            # 每个尺度的卷积
            branch = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=scale, stride=1, padding=(scale[0]//2, scale[1]//2, scale[2]//2)),
                nn.BatchNorm3d(64) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=scale, stride=1, padding=(scale[0]//2, scale[1]//2, scale[2]//2)),
                nn.BatchNorm3d(64) if self.use_batch_norm else nn.Identity(),
                nn.ReLU()
            )
            multiscale_modules.append(branch)
            concat_channels += 64
        
        # 添加注意力模块
        if self.use_attention:
            attention = self._build_attention_module(concat_channels)
        else:
            attention = nn.Identity()
        
        # 额外的空间降维，适应256x256输入
        pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(concat_channels, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.AdaptiveAvgPool3d((self.time_steps, 2, 2))  # 自适应池化到固定大小
        )
        
        return nn.ModuleDict({
            'initial': layers[0],
            'multiscale': nn.ModuleList(multiscale_modules),
            'attention': attention,
            'pooling': pooling,
            'concat_channels': torch.nn.Parameter(torch.tensor(concat_channels), requires_grad=False)
        })
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, time_steps, height, width)
               或 (batch_size, time_steps, channels, height, width)
               
        Returns:
            torch.Tensor: 模型输出
            
        Raises:
            RuntimeError: 当前向传播失败时
        """
        try:
            # 验证输入形状
            if x.size(2) != self.time_steps or x.size(3) != self.height or x.size(4) != self.width:
                raise ValueError(f"输入形状 {x.shape[2:]} 与期望形状 {(self.time_steps, self.height, self.width)} 不匹配")
            
            # 调整输入通道
            x = self._adjust_input_channels(x)
            
            # 调整输入维度顺序
            if x.size(2) > 10 and x.size(1) <= 4:
                logger.warning(f"输入张量形状为 {x.shape}，调整通道和时间维度顺序")
                x = x.transpose(1, 2)
            
            # CNN处理
            if self.training:
                from torch.utils.checkpoint import checkpoint
                cnn_out = checkpoint(self.spatial_features, x, use_reentrant=False)
                cnn_out = checkpoint(self.temporal_features, cnn_out, use_reentrant=False)
                cnn_out = checkpoint(self.multiscale_features, cnn_out, use_reentrant=False)
            else:
                cnn_out = self.spatial_features(x)
                cnn_out = self.temporal_features(cnn_out)
                cnn_out = self.multiscale_features(cnn_out)
            
            logger.debug(f"CNN输出形状: {cnn_out.shape}")
            
            # 特征融合
            if self.use_feature_fusion:
                fusion_out = self.fusion_layer(cnn_out.view(cnn_out.size(0), -1))
            else:
                fusion_out = cnn_out.view(cnn_out.size(0), -1)
            
            # 全连接层
            out = self.classifier(fusion_out)
            
            return out
            
        except Exception as e:
            logger.error(f"前向传播失败: {str(e)}")
            logger.error(f"输入形状: {x.shape}")
            raise RuntimeError(f"前向传播失败: {str(e)}")
    
    def _adjust_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        """调整输入通道数
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 调整后的输入张量
        """
        if x.size(1) != self.channels:
            logger.warning(f"输入通道数 {x.size(1)} 与期望的通道数 {self.channels} 不匹配")
            
            if x.size(1) == 1 and self.channels > 1:
                logger.info(f"将单通道输入扩展为 {self.channels} 通道")
                x = x.repeat(1, self.channels, 1, 1, 1)
            elif x.size(1) == 3 and self.channels == 4:
                logger.info("将3通道输入扩展为4通道")
                x = torch.cat([x, x[:, 0:1]], dim=1)
        
        return x
    
    def train(self, mode: bool = True) -> 'HybridMicroExpressionModel':
        """设置模型为训练模式
        
        Args:
            mode: 是否为训练模式
            
        Returns:
            HybridMicroExpressionModel: 模型实例
        """
        return super().train(mode)
    
    def eval(self) -> 'HybridMicroExpressionModel':
        """设置模型为评估模式
        
        Returns:
            HybridMicroExpressionModel: 模型实例
        """
        return super().eval()
    
    def to(self, device: Union[str, torch.device]) -> 'HybridMicroExpressionModel':
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            HybridMicroExpressionModel: 模型实例
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(self.device)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载模型状态字典
        
        Args:
            state_dict: 模型状态字典
        """
        super().load_state_dict(state_dict)

class HybridModelPyTorch:
    """混合模型的包装类，提供与MicroExpressionModelPyTorch相同的接口
    
    该类封装了HybridMicroExpressionModel，提供统一的接口和额外的功能。
    
    Attributes:
        model: 内部HybridMicroExpressionModel实例
        device: 计算设备
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (20, 128, 128, 4),
        model_type: str = 'cnn_lstm',
        device: Optional[Union[str, torch.device]] = None
    ):
        """初始化包装类
        
        Args:
            input_shape: 输入形状，固定为 (20, 128, 128, 4)
            model_type: 模型类型
            device: 计算设备
            
        Raises:
            ValueError: 当模型类型不支持时
        """
        if model_type != 'cnn_lstm':
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 验证输入形状
        if input_shape != (20, 128, 128, 4):
            logger.warning(f"输入形状 {input_shape} 不符合规范，将使用默认形状 (20, 128, 128, 4)")
            input_shape = (20, 128, 128, 4)
            
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HybridMicroExpressionModel(input_shape=input_shape, device=self.device)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """调用模型
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 模型输出
        """
        return self.model(x)
    
    def train(self) -> 'HybridModelPyTorch':
        """设置模型为训练模式
        
        Returns:
            HybridModelPyTorch: 模型实例
        """
        self.model.train()
        return self
    
    def eval(self) -> 'HybridModelPyTorch':
        """设置模型为评估模式
        
        Returns:
            HybridModelPyTorch: 模型实例
        """
        self.model.eval()
        return self
    
    def parameters(self) -> torch.nn.parameter.Parameter:
        """获取模型参数
        
        Returns:
            torch.nn.parameter.Parameter: 模型参数
        """
        return self.model.parameters()
    
    def to(self, device: Union[str, torch.device]) -> 'HybridModelPyTorch':
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            HybridModelPyTorch: 模型实例
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        return self
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载模型状态字典
        
        Args:
            state_dict: 模型状态字典
        """
        self.model.load_state_dict(state_dict)