import torch
import torch.nn as nn
import torch.nn.functional as F

# 从utils导入NUM_CLASSES
from .utils import NUM_CLASSES, logger

# --- 第一阶段模型：单帧分类用CNN ---
class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_channels=1):
        """
        输入应为 (B, C, H, W)，例如 (B, 1, 128, 128)
        """
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        # 第一块
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding='same') # 原始为5x5
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第四块
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 展平和全连接层
        self.flatten = nn.Flatten()
        # 动态计算展平后的大小（假设输入为128x128）
        # 经过4次池化(2x2, stride=2)，128 -> 64 -> 32 -> 16 -> 8
        flattened_size = 512 * 8 * 8 
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU()
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        # Softmax通常在损失函数（CrossEntropyLoss）中实现
    def forward(self, x, return_features=False):
        """
        参数：
            x (torch.Tensor): 输入张量 (B, C, H, W)
            return_features (bool): 若为True，则返回最终分类器前的特征。
                                   用于作为LSTM的backbone时。
        返回：
            torch.Tensor: 输出logits (B, num_classes) 或特征 (B, feature_dim)
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.drop3(self.relu3(self.conv3(x))))
        x = self.pool4(self.drop4(self.relu4(self.conv4(x))))
        
        x = self.flatten(x)
        features = self.relu_fc1(self.fc1(x)) # 第一层全连接后的特征
        if return_features:
            return features # 返回形状 (B, 256)
        x = self.drop_fc1(features)
        x = self.fc2(x) # logits
        return x

# --- 注意力模块（可选，模仿原项目） ---
class AttentionBlock(nn.Module):
    def __init__(self, feature_dim, attention_hidden_dim=None):
        """
        用于序列数据的简单注意力机制。
        参数：
            feature_dim (int): 每个时间步的特征维度 (F)
            attention_hidden_dim (int, optional): 注意力隐藏层维度。默认feature_dim // 2。
        """
        super(AttentionBlock, self).__init__()
        if attention_hidden_dim is None:
            attention_hidden_dim = feature_dim // 2 if feature_dim // 2 > 0 else 1 # 保证为正

        # 计算注意力分数的层
        self.attention_layer1 = nn.Linear(feature_dim, attention_hidden_dim)
        self.tanh = nn.Tanh()
        self.attention_layer2 = nn.Linear(attention_hidden_dim, 1) # 每个时间步输出一个分数
        
    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 输入张量 (B, T, F)
        返回：
            torch.Tensor: 应用注意力后的序列 (B, T, F)
        """
        # 输入形状: (B, T, F)
        # 计算注意力分数
        attn_hidden = self.tanh(self.attention_layer1(x)) # (B, T, attention_hidden_dim)
        attn_logits = self.attention_layer2(attn_hidden)   # (B, T, 1)

        # 计算注意力权重（概率）
        attn_weights = F.softmax(attn_logits, dim=1)  # 对时间维度T做softmax -> (B, T, 1)

        # 应用权重到输入序列
        # 加权和：attn_weights (B, T, 1) * x (B, T, F) -> (B, T, F)
        weighted_sequence = x * attn_weights

        # 原始代码返回加权序列
        # 若需单一上下文向量，可用 torch.sum(weighted_sequence, dim=1)

        return weighted_sequence # 返回加权后的序列
        
# --- 第二阶段模型：仅LSTM（使用预提取特征） ---
class LSTMOnlyModel(nn.Module):
    def __init__(self, 
                 input_feature_dim, # 来自CNN的特征维度
                 lstm_hidden_size=128, 
                 lstm_num_layers=1, 
                 num_classes=NUM_CLASSES, 
                 sequence_length=64, # 填充/截断后的固定序列长度
                 use_attention=True, 
                 dropout_lstm=0.3, 
                 dropout_fc=0.5):
        """
        使用预提取特征的序列分类LSTM模型。
        输入形状: (B, T, input_feature_dim)
        """
        super(LSTMOnlyModel, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length # 若用AttentionBlock时需要
        self.lstm_hidden_size = lstm_hidden_size
        self.use_attention = use_attention

        # 1. LSTM层
        self.lstm1 = nn.LSTM(input_size=input_feature_dim, 
                             hidden_size=lstm_hidden_size, 
                             num_layers=lstm_num_layers,
                             batch_first=True, 
                             bidirectional=False)

        # 2. 可选注意力层
        if self.use_attention:
            self.attention = AttentionBlock(feature_dim=lstm_hidden_size)
        
        # 3. 第二层LSTM
        lstm2_input_size = lstm_hidden_size
        self.lstm2 = nn.LSTM(input_size=lstm2_input_size,
                             hidden_size=lstm_hidden_size // 2,
                             num_layers=lstm_num_layers,
                             batch_first=True, 
                             bidirectional=False)
                             
        self.dropout_lstm_out = nn.Dropout(dropout_lstm)

        # 4. 最终全连接层
        fc1_input_size = lstm_hidden_size // 2
        self.fc1 = nn.Linear(fc1_input_size, fc1_input_size // 2) # 例如64->32
        self.dropout_fc1 = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(fc1_input_size // 2, num_classes)

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 输入特征序列张量 (B, T, input_feature_dim)
                              T应为目标序列长度
        返回：
            torch.Tensor: 输出logits (B, num_classes)
        """
        # 输入形状: (B, T, input_feature_dim)
        
        # 1. 通过第一层LSTM
        lstm1_out, _ = self.lstm1(x) # 形状 (B, T, lstm_hidden_size)
        
        # 2. 可选注意力
        if self.use_attention:
             lstm1_out = self.attention(lstm1_out) # 形状 (B, T, lstm_hidden_size)
             
        # 3. 通过第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out) # 形状 (B, T, lstm_hidden_size // 2)
        
        # --- 新增：选取最后一个时间步的输出 ---
        # lstm2_out 形状 (Batch, Seq_len, Features)
        lstm_last_step_out = lstm2_out[:, -1, :] # 形状 (B, lstm_hidden_size // 2)

        # 对最后一个时间步的输出应用Dropout
        lstm_final_out = self.dropout_lstm_out(lstm_last_step_out)
        
        # 后续全连接层处理 (B, Features) 形状输入
        dense_hidden = self.fc1(lstm_final_out) # 形状 (B, fc1_input_size // 2)
        dense_out = self.dropout_fc1(dense_hidden)
        logits = self.fc2(dense_out)
        
        return logits

# --- 第二阶段模型：CNN+LSTM ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, 
                 cnn_feature_dim=256, # CNNModel特征层输出维度
                 lstm_hidden_size=128, 
                 lstm_num_layers=1, # 原始等效于2层LSTM
                 num_classes=NUM_CLASSES, 
                 sequence_length=32,
                 use_attention=True, 
                 dropout_lstm=0.3, 
                 dropout_fc=0.5):
        """
        用于序列分类的CNN+LSTM模型。
        假设CNN部分已预训练并冻结。
        """
        super(CNN_LSTM_Model, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.use_attention = use_attention

        # 1. CNN骨干网络（需后续加载权重并冻结）
        # 这里只创建实例，权重需后续加载
        self.cnn_backbone = CNNModel(num_classes=num_classes) # num_classes无关紧要
        # 后续调用cnn_backbone(x, return_features=True)
        
        # 2. LSTM层
        # LSTM输入为CNN特征，形状 (B, T, cnn_feature_dim)
        self.lstm1 = nn.LSTM(input_size=cnn_feature_dim, 
                             hidden_size=lstm_hidden_size, 
                             num_layers=lstm_num_layers, # 第一层LSTM
                             batch_first=True, 
                             bidirectional=False) # 与原始一致

        # 3. 可选注意力层
        if self.use_attention:
            self.attention = AttentionBlock(feature_dim=lstm_hidden_size)
        
        # 4. 第二层LSTM（如原始）
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_size, # LSTM1/Attention输出
                             hidden_size=lstm_hidden_size // 2, # 原始用64
                             num_layers=lstm_num_layers,
                             batch_first=True, 
                             bidirectional=False)
                             
        self.dropout_lstm_out = nn.Dropout(dropout_lstm) # LSTM后Dropout

        # 5. 最终全连接层（如原始）
        self.fc1 = nn.Linear(lstm_hidden_size // 2, lstm_hidden_size // 4) # 原始用32
        self.dropout_fc1 = nn.Dropout(dropout_fc) # 原始用0.5
        self.fc2 = nn.Linear(lstm_hidden_size // 4, num_classes)

    def load_cnn_backbone(self, checkpoint_path, freeze=True):
        """
        从第一阶段训练的checkpoint加载权重到self.cnn_backbone。
        参数：
            checkpoint_path (str): .pth文件路径
            freeze (bool): 若为True，则冻结CNN骨干网络权重
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 尝试直接加载state_dict或'model_state_dict'键
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # 加载权重到骨干网络
            self.cnn_backbone.load_state_dict(state_dict)
            logger.info(f"成功加载CNN骨干权重: {checkpoint_path}")
            
            if freeze:
                for param in self.cnn_backbone.parameters():
                    param.requires_grad = False
                self.cnn_backbone.eval() # 设为评估模式
                logger.info("CNN骨干已冻结并设为eval模式。")
                
        except FileNotFoundError:
            logger.error(f"未找到CNN骨干checkpoint: {checkpoint_path}。LSTM阶段需要预训练CNN。")
            raise
        except Exception as e:
            logger.error(f"加载CNN骨干权重出错: {checkpoint_path}: {e}")
            raise

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 输入序列张量 (B, T, C, H, W)
        返回：
            torch.Tensor: 输出logits (B, num_classes)
        """
        batch_size, time_steps, C, H, W = x.size()
        
        # 1. 对每帧应用CNN骨干
        # 输入重塑: (B, T, C, H, W) -> (B*T, C, H, W)
        cnn_in = x.view(batch_size * time_steps, C, H, W)
        # 骨干输出特征: (B*T, C, H, W) -> (B*T, cnn_feature_dim)
        cnn_out_features = self.cnn_backbone(cnn_in, return_features=True)
        # 恢复为序列: (B*T, cnn_feature_dim) -> (B, T, cnn_feature_dim)
        lstm_in = cnn_out_features.view(batch_size, time_steps, -1)
        
        # 2. 通过第一层LSTM
        lstm1_out, _ = self.lstm1(lstm_in) # 形状 (B, T, lstm_hidden_size)
        
        # 3. 可选注意力
        if self.use_attention:
             lstm1_out = self.attention(lstm1_out) # 形状 (B, T, lstm_hidden_size)
             
        # 4. 通过第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out) # 形状 (B, T, lstm_hidden_size // 2)
        
        # --- 修改：同样仅使用最后一个时间步 --- 
        lstm_last_step_out = lstm2_out[:, -1, :] # 形状 (B, lstm_hidden_size // 2)
        
        x = self.dropout_lstm_out(lstm_last_step_out)
        
        # 5. 最终全连接层
        dense_hidden = self.fc1(x) # 形状 (B, lstm_hidden_size // 4)
        x = self.dropout_fc1(dense_hidden)
        x = self.fc2(x)
        
        return x


# 用于结构测试的示例
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- 测试CNNModel ---")
    cnn_model_test = CNNModel(num_classes=7).to(device)
    dummy_frame = torch.randn(4, 1, 128, 128).to(device) # B, C, H, W
    try:
        # 测试分类输出
        output_logits = cnn_model_test(dummy_frame)
        print(f"CNN Logits输出形状: {output_logits.shape}")
        assert output_logits.shape == (4, 7)
        # 测试特征输出
        output_features = cnn_model_test(dummy_frame, return_features=True)
        print(f"CNN特征输出形状: {output_features.shape}")
        assert output_features.shape[1] == 256 # 检查特征维度
        print("CNNModel测试通过。")
    except Exception as e:
        print(f"CNNModel测试出错: {e}")

    print("\n--- 测试LSTMOnlyModel ---")
    lstm_only_model_test = LSTMOnlyModel(input_feature_dim=256, sequence_length=64).to(device)
    dummy_features = torch.randn(4, 64, 256).to(device) # B, T, F
    try:
        output_logits_lstm = lstm_only_model_test(dummy_features)
        print(f"LSTMOnly Logits输出形状: {output_logits_lstm.shape}")
        assert output_logits_lstm.shape == (4, NUM_CLASSES)
        print("LSTMOnlyModel测试通过。")
    except Exception as e:
        print(f"LSTMOnlyModel测试出错: {e}")

    print("\n--- 测试CNN_LSTM_Model ---")
    cnn_lstm_model_test = CNN_LSTM_Model(num_classes=7, sequence_length=16, lstm_hidden_size=128).to(device)
    dummy_sequence = torch.randn(4, 16, 1, 128, 128).to(device) # B, T, C, H, W
    
    # 通常应先加载预训练权重再forward
    # cnn_lstm_model_test.load_cnn_backbone('path/to/stage1_cnn.pth')
    # 此处测试跳过加载，骨干未训练
    
    try:
        output_seq_logits = cnn_lstm_model_test(dummy_sequence)
        print(f"CNN_LSTM Logits输出形状: {output_seq_logits.shape}")
        assert output_seq_logits.shape == (4, 7)
        print("CNN_LSTM_Model前向测试通过（未加载权重）。")
        
        # 若冻结骨干网络，仅LSTM/FC参数需训练（需先加载）
        # cnn_lstm_model_test.load_cnn_backbone('dummy.pth', freeze=True) # 假设dummy文件存在
        # trainable_params = sum(p.numel() for p in cnn_lstm_model_test.parameters() if p.requires_grad)
        # print(f"冻结骨干后可训练参数量: {trainable_params}")
        
    except Exception as e:
        print(f"CNN_LSTM_Model测试出错: {e}")


# ============================================================
# 新增：ROI 光流 Transformer 分类器
# 参考 MEGC2024 STR 第2名方案 (Transformer_rois)
# 作为新模型类型，不影响现有 CNN/LSTM/CNN_LSTM 架构
# ============================================================

class ROITransformerModel(nn.Module):
    """
    基于 ROI 光流特征的小型 Transformer 微表情分类器

    参考: MEGC2024 STR 第2名 (HIT) — 18-ROI 光流 + Transformer
    本实现适配 7-ROI（与 OpticalFlowSpottingEngine 一致），参数更轻量。

    输入: (B, T, num_rois * 2) — T帧序列，每帧 num_rois 个 ROI 的 (dx, dy) 光流
    输出: (B, num_classes) — 分类 logits
    """

    def __init__(self,
                 num_rois: int = 7,
                 max_seq_len: int = 64,
                 num_classes: int = NUM_CLASSES,
                 dim: int = 64,
                 depth: int = 3,
                 heads: int = 4,
                 mlp_dim: int = 128,
                 dim_head: int = 16,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1):
        super(ROITransformerModel, self).__init__()

        self.num_rois = num_rois
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        patch_dim = num_rois * 2  # 每帧输入维度: 7 ROI × 2 (dx, dy) = 14

        # 输入投影: patch_dim -> dim
        self.input_proj = nn.Linear(patch_dim, dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 位置编码 (可学习)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN（更稳定）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, num_rois * 2) 光流特征序列
        Returns:
            (B, num_classes) logits
        """
        b, t, _ = x.shape

        # 投影到 dim 维
        x = self.input_proj(x)  # (B, T, dim)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, dim)

        # 添加位置编码（截断或填充到实际长度）
        x = x + self.pos_embedding[:, :t + 1, :]
        x = self.emb_dropout(x)

        # Transformer 编码
        x = self.transformer(x)  # (B, T+1, dim)

        # 取 CLS token 输出
        cls_out = x[:, 0]  # (B, dim)
        cls_out = self.norm(cls_out)

        # 分类
        return self.mlp_head(cls_out)  # (B, num_classes)

    def count_parameters(self):
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)