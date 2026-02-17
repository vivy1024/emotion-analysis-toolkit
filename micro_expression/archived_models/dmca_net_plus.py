class DMCANetEnhanced(nn.Module):
    """增强版的动态多通道注意力网络(Dynamic Multi-Channel Attention Network)"""
    
    @classmethod
    def create_for_device(cls, input_shape, output_shape, device):
        """根据设备自动配置模型参数"""
        frames, height, width, channels = input_shape
        
        # 获取设备内存信息
        if device.type == 'cuda':
            # GPU配置
            mem_info = torch.cuda.get_device_properties(device).total_memory
            mem_gb = mem_info / (1024**3)
            
            if mem_gb < 4:  # 小于4GB内存
                return cls(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    feature_dim=64,  # 减少特征维度
                    num_attention_heads=1,  # 减少注意力头数
                    use_temporal_attention=False,  # 禁用时序注意力
                    memory_efficient=True
                )
            elif mem_gb < 8:  # 4-8GB内存
                return cls(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    feature_dim=128,
                    num_attention_heads=2,
                    memory_efficient=True
                )
            else:  # 8GB以上内存
                return cls(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    feature_dim=256,
                    num_attention_heads=4,
                    use_temporal_attention=True,
                    use_diagonal_attention=True
                )
        else:
            # CPU配置 - 保守设置
            return cls(
                input_shape=input_shape,
                output_shape=output_shape,
                feature_dim=64,
                num_attention_heads=1,
                use_temporal_attention=False,
                memory_efficient=True
            )
    
    def __init__(self, 
                 input_shape=(20, 256, 256, 4),
                 output_shape=7,
                 feature_dim=256,
                 num_attention_heads=4,
                 dropout_rate=0.5,
                 use_temporal_attention=True,
                 use_spatial_attention=True,
                 use_channel_attention=True,
                 use_diagonal_attention=False,
                 use_adversarial=False,
                 memory_efficient=False,
                 attention_weight=1.0):  # 增加注意力权重参数
        """
        初始化增强版DMCA网络
        
        Args:
            input_shape: 输入形状(frames, height, width, channels)
            output_shape: 输出类别数
            feature_dim: 特征维度
            num_attention_heads: 注意力头数
            dropout_rate: Dropout比率
            use_temporal_attention: 是否使用时序注意力
            use_spatial_attention: 是否使用空间注意力
            use_channel_attention: 是否使用通道注意力
            use_diagonal_attention: 是否使用对角注意力
            use_adversarial: 是否使用对抗训练
            memory_efficient: 是否使用内存高效模式
            attention_weight: 注意力权重，用于放大或缩小注意力机制的影响
        """
        super(DMCANetEnhanced, self).__init__()
        
        self.frames, self.height, self.width, self.channels = input_shape
        self.output_shape = output_shape
        self.feature_dim = feature_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.use_temporal_attention = use_temporal_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        self.use_diagonal_attention = use_diagonal_attention
        self.use_adversarial = use_adversarial
        self.memory_efficient = memory_efficient
        self.attention_weight = attention_weight
        
        # 检查输入通道数，支持OpenFace增强特征
        if self.channels > 4:
            self.has_openface_features = True
            self.openface_channels = self.channels - 4
            print(f"检测到OpenFace特征通道: {self.openface_channels}个额外通道")
        else:
            self.has_openface_features = False
            self.openface_channels = 0
        
        # 特征提取层
        # 初始卷积层 - 处理输入数据
        self.conv_init = nn.Sequential(
            nn.Conv3d(self.channels, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # 增加OpenFace特征处理路径
        if self.has_openface_features:
            # 单独处理OpenFace特征
            self.openface_feature_extractor = nn.Sequential(
                nn.Conv3d(self.openface_channels, 32, kernel_size=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            )
            
            # 融合原始特征和OpenFace特征
            self.feature_fusion = nn.Sequential(
                nn.Conv3d(64 + 64, 96, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True)
            )
            
            # 更新后续层的输入通道数
            conv1_in_channels = 96
        else:
            conv1_in_channels = 64
        
        # 主干卷积层
        self.conv1 = nn.Sequential(
            nn.Conv3d(conv1_in_channels, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, feature_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # 计算卷积后的特征维度
        conv_frames = self.frames // 4
        conv_height = self.height // 8
        conv_width = self.width // 8
        
        # 注意力机制
        if use_temporal_attention:
            self.temporal_attention = MultiHeadAttention(
                feature_dim, num_attention_heads, dropout_rate
            )
        
        if use_spatial_attention:
            self.spatial_attention = MultiHeadAttention(
                feature_dim, num_attention_heads, dropout_rate
            )
        
        if use_channel_attention:
            self.channel_attention = ChannelAttention(feature_dim)
        
        if use_diagonal_attention:
            self.diagonal_attention = DiagonalAttention(feature_dim)
        
        # 全局池化层
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 全连接分类器
        fc_input_dim = feature_dim
        
        if memory_efficient:
            # 内存高效版本：直接使用池化后的特征
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, output_shape)
            )
        else:
            # 更大版本：使用更多全连接层
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, output_shape)
            )
        
        # 对抗性鉴别器（如果启用）
        if use_adversarial:
            self.discriminator = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, frames, height, width, channels)
            return_features: 是否返回特征向量（用于对抗训练）
            
        Returns:
            output: 分类输出
            features: （可选）特征向量
        """
        # 重新排列输入维度为(batch_size, channels, frames, height, width)
        x = x.permute(0, 4, 1, 2, 3)
        
        # 分离基础特征和OpenFace特征（如果有）
        if self.has_openface_features:
            # 基础特征(前4通道)和OpenFace特征(额外通道)
            base_features = x[:, :4]
            openface_features = x[:, 4:]
            
            # 并行处理两组特征
            x_base = self.conv_init(base_features)
            x_openface = self.openface_feature_extractor(openface_features)
            
            # 确保两组特征的空间尺寸匹配
            if x_base.size(3) != x_openface.size(3) or x_base.size(4) != x_openface.size(4):
                x_openface = F.interpolate(
                    x_openface, 
                    size=(x_openface.size(2), x_base.size(3), x_base.size(4)),
                    mode='nearest'
                )
            
            # 特征融合
            x = torch.cat([x_base, x_openface], dim=1)
            x = self.feature_fusion(x)
        else:
            # 只处理基础特征
            x = self.conv_init(x)
        
        # 卷积特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 保存卷积后的原始特征
        original_features = x
        
        # 应用注意力机制
        if self.use_temporal_attention:
            # 重新排列为(batch, frames, features)
            batch, channels, frames, height, width = x.size()
            features_temp = x.permute(0, 2, 1, 3, 4).reshape(batch, frames, -1)
            
            # 应用时序注意力
            features_temp = self.temporal_attention(features_temp, features_temp, features_temp)
            
            # 恢复原始形状
            features_temp = features_temp.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
            
            # 应用注意力权重
            x = x + self.attention_weight * features_temp
        
        if self.use_spatial_attention:
            # 重新排列为(batch, spatial_points, features)
            batch, channels, frames, height, width = x.size()
            features_spatial = x.permute(0, 3, 4, 1, 2).reshape(batch, height * width, -1)
            
            # 应用空间注意力
            features_spatial = self.spatial_attention(features_spatial, features_spatial, features_spatial)
            
            # 恢复原始形状
            features_spatial = features_spatial.reshape(batch, height, width, channels, frames).permute(0, 3, 4, 1, 2)
            
            # 应用注意力权重
            x = x + self.attention_weight * features_spatial
        
        if self.use_channel_attention:
            # 应用通道注意力
            channel_weights = self.channel_attention(x)
            x = x * channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        if self.use_diagonal_attention:
            # 应用对角注意力（捕捉通道间依赖关系）
            x = self.diagonal_attention(x)
        
        # 全局池化
        x = self.global_avg_pool(x)
        
        # 获取特征向量
        features = x.view(x.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        if return_features:
            return output, features
        else:
            return output

class DiagonalAttention(nn.Module):
    """对角注意力机制 - 捕捉特征维度间的对角关系"""
    
    def __init__(self, channels, reduction_ratio=16):
        super(DiagonalAttention, self).__init__()
        
        self.channels = channels
        
        # 特征降维
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        # 特征升维
        self.fc2 = nn.Linear(channels // reduction_ratio, channels * 2)  # 输出2倍通道，一半用于缩放，一半用于位移
        
    def forward(self, x):
        # 输入 x: (batch, channels, frames, height, width)
        batch_size, channels, frames, height, width = x.size()
        
        # 全局池化
        y = torch.mean(x.view(batch_size, channels, -1), dim=2)  # (batch, channels)
        
        # 通过全连接层
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        
        # 分离缩放和位移参数
        scale, shift = torch.chunk(y, 2, dim=1)
        
        # Sigmoid和Tanh激活
        scale = torch.sigmoid(scale).view(batch_size, channels, 1, 1, 1) * 2.0  # 范围[0, 2]
        shift = torch.tanh(shift).view(batch_size, channels, 1, 1, 1) * 0.1     # 范围[-0.1, 0.1]
        
        # 应用对角变换: y = scale * x + shift
        return scale * x + shift 