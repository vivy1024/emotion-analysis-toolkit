"""
CNN模型模块 - 提供基础CNN模型实现

该模块提供了基于PyTorch的基础CNN微表情识别模型实现。
"""

import logging
from typing import Tuple, Optional, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroExpressionModelPyTorch(nn.Module):
    """基于PyTorch的微表情识别CNN模型
    
    该模型使用基础的3D CNN架构，适合作为基准模型使用。
    
    Attributes:
        device: 计算设备
        time_steps: 时间步数
        height: 图像高度
        width: 图像宽度
        channels: 输入通道数
        expected_channels: 期望的输入通道数
        dropout_rate: 中间层dropout率
        final_dropout_rate: 最终层dropout率
        conv_output_size: 卷积层输出大小
        weight_decay: L2正则化系数
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (20, 256, 256, 4),  # 更新默认输入大小为256x256
        output_shape: int = 7,  # 输出类别数
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = False,
        use_residual: bool = False,
        use_feature_fusion: bool = False,
        device: Union[str, torch.device] = 'cuda'
    ):
        """初始化模型
        
        Args:
            input_shape: 输入形状，格式为 (time_steps, height, width, channels)
            output_shape: 输出类别数
            use_batch_norm: 是否使用批量标准化
            dropout_rate: dropout率
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
            use_feature_fusion: 是否使用特征融合
            device: 计算设备
            
        Raises:
            ValueError: 当输入形状无效时
            RuntimeError: 当CUDA不可用时
        """
        super(MicroExpressionModelPyTorch, self).__init__()
        
        # 检查CUDA是否可用
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，回退到CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        self.debug = True  # 启用调试输出
        self._first_forward = True  # 跟踪是否是首次前向传播
        
        # 保存配置
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_feature_fusion = use_feature_fusion
        
        # 解析输入形状
        if len(input_shape) == 4:
            self.time_steps, self.height, self.width, self.channels = input_shape
        elif len(input_shape) == 3:
            self.time_steps, self.height, self.width = input_shape
            self.channels = 1
        else:
            raise ValueError(f"输入形状必须是3维或4维，但得到了{len(input_shape)}维")
        
        logger.info(f"模型初始化 - 输入形状: time_steps={self.time_steps}, height={self.height}, width={self.width}, channels={self.channels}")
        
        # 期望4通道输入 (灰度+光流XY+LBP纹理)
        self.expected_channels = 4
        
        # 检查通道数是否与期望一致
        if self.channels != self.expected_channels:
            logger.warning(f"输入通道数 {self.channels} 与期望的通道数 {self.expected_channels} 不匹配")
            # 修正 channels 为 4，确保模型可以接受4通道输入
            self.channels = self.expected_channels
        
        # 增加dropout率以减少过拟合
        self.final_dropout_rate = self.dropout_rate * 0.6
        
        # 构建卷积层
        self.conv_layers = self._build_basic_conv_layers()
        logger.info("使用基础CNN架构")
        
        # 计算卷积层输出大小
        try:
            with torch.no_grad():
                input_tensor = torch.zeros(1, self.channels, self.time_steps, self.height, self.width)
                logger.info(f"测试卷积输出尺寸 - 输入形状: {input_tensor.shape}")
                
                # 逐层跟踪输出尺寸变化
                x = input_tensor
                for i, layer in enumerate(self.conv_layers):
                    x = layer(x)
                    if isinstance(layer, (nn.Conv3d, nn.MaxPool3d)):
                        logger.info(f"层 {i} ({layer.__class__.__name__}) 后的形状: {x.shape}")
                
                self.conv_output_size = x.view(1, -1).size(1)
                logger.info(f"最终CNN输出尺寸: {x.shape}, 展平后: {self.conv_output_size}")
        except Exception as e:
            logger.error(f"计算CNN输出尺寸时出错: {str(e)}")
            # 为全连接层设置默认大小，256x256输入大小的预估值
            self.conv_output_size = 98304  # 为(20,256,256,4)输入设置的预估值，是128x128的四倍
            logger.warning(f"使用预估的CNN输出尺寸: {self.conv_output_size}")
        
        # 全连接层 - 针对更大的输入调整架构
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 1024),  # 增加节点数以适应更大的输入
            nn.BatchNorm1d(1024) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.final_dropout_rate),
            nn.Linear(256, output_shape)  # 使用output_shape参数决定输出类别数
        )
        
        # L2正则化
        self.weight_decay = 0.001
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        logger.info(f"成功创建基础CNN模型，输入形状: {input_shape}, 全连接层输入尺寸: {self.conv_output_size}")
    
    def _build_basic_conv_layers(self) -> nn.Sequential:
        """构建基础卷积层，添加更多正则化，并优化以处理256x256输入
        
        Returns:
            nn.Sequential: 基础卷积层序列
        """
        layers = []
        
        # 第一个卷积块 - 使用更小的步长
        layers.extend([
            nn.Conv3d(self.channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout3d(0.2) if self.dropout_rate > 0 else nn.Identity(),
            # 使用更大的池化核心和步长，减少特征图尺寸
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])
        
        # 第二个卷积块
        layers.extend([
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout3d(0.2) if self.dropout_rate > 0 else nn.Identity(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])
        
        # 第三个卷积块
        layers.extend([
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout3d(0.2) if self.dropout_rate > 0 else nn.Identity(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])
        
        # 针对256x256输入添加第四个卷积块，进一步降维
        layers.extend([
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout3d(0.2) if self.dropout_rate > 0 else nn.Identity(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, time_steps, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, 7)
        """
        try:
            batch_size = x.size(0)
            
            # 检查输入形状正确性
            if x.dim() != 5:
                raise ValueError(f"输入形状必须是5维，但得到了{x.dim()}维")
            
            # 用于调试
            if self.debug and self._first_forward:
                logger.info(f"输入形状: {x.shape}")
            
            # CNN卷积层处理
            x = self.conv_layers(x)
            
            if self.debug and self._first_forward:
                logger.info(f"CNN输出形状: {x.shape}")
            
            # 检查是否有NaN值
            if torch.isnan(x).any():
                logger.error("CNN输出包含NaN值")
                
            # 展平CNN输出以供全连接层使用
            conv_out_flat_size = x.view(batch_size, -1).size(1)
            
            # 自适应批处理以降低内存消耗
            if batch_size > 1 and conv_out_flat_size > 100000:
                # 通过小批量处理减少内存使用
                outputs = []
                sub_batch_size = max(1, batch_size // 4)  # 更小的子批次大小
                
                for i in range(0, batch_size, sub_batch_size):
                    end_idx = min(i + sub_batch_size, batch_size)
                    sub_batch = x[i:end_idx]
                    
                    # 展平并通过全连接层处理
                    sub_batch_flat = sub_batch.view(end_idx - i, -1)
                    sub_output = self.fc(sub_batch_flat)
                    outputs.append(sub_output)
                
                # 合并子批次输出
                x = torch.cat(outputs, dim=0)
            else:
                # 如果批次很小或展平尺寸可接受，正常处理
                x = x.view(batch_size, -1)
                x = self.fc(x)
                
            # 设置为非首次前向传播
            if self._first_forward:
                self._first_forward = False
                logger.info("形状信息已记录，后续前向传播将不再显示形状信息")
                
            return x
            
        except Exception as e:
            logger.error(f"前向传播出错: {str(e)}")
            raise
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """调用模型
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 模型输出
        """
        return self.forward(x)
    
    def train(self, mode: bool = True) -> 'MicroExpressionModelPyTorch':
        """设置模型为训练模式
        
        Args:
            mode: 是否为训练模式
            
        Returns:
            MicroExpressionModelPyTorch: 模型实例
        """
        return super().train(mode)
    
    def eval(self) -> 'MicroExpressionModelPyTorch':
        """设置模型为评估模式
        
        Returns:
            MicroExpressionModelPyTorch: 模型实例
        """
        return super().eval()
    
    def parameters(self) -> torch.nn.parameter.Parameter:
        """获取模型参数
        
        Returns:
            torch.nn.parameter.Parameter: 模型参数
        """
        return super().parameters()
    
    def to(self, device: Union[str, torch.device]) -> 'MicroExpressionModelPyTorch':
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            MicroExpressionModelPyTorch: 模型实例
            
        Raises:
            RuntimeError: 当尝试使用非CUDA设备时
        """
        # 强制使用CUDA
        device_str = str(device).lower()
        if 'cuda' not in device_str:
            raise RuntimeError("模型只能在CUDA设备上运行")
            
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(self.device)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载模型状态字典
        
        Args:
            state_dict: 模型状态字典
        """
        super().load_state_dict(state_dict)
        
    def reset_debug_flags(self):
        """重置调试标志
        
        用于在需要查看形状信息时手动调用，例如调试模型结构问题时
        """
        self._first_forward = True
        logger.info("已重置形状信息调试标志，下一次前向传播将显示形状信息")