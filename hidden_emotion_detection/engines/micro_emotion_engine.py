#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微表情分析引擎
基于3D-CNN深度学习模型实现7类微表情识别，处理连续帧序列
"""

import os
import logging
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import threading
from collections import deque, defaultdict, Counter

from hidden_emotion_detection.config import ConfigManager, config_manager as global_config_manager # 新的导入
from hidden_emotion_detection.core.data_types import EmotionResult, EmotionType, FaceBox, Event, EventType, FaceDetection
from hidden_emotion_detection.core.event_bus import EventBus
from hidden_emotion_detection.utils.decorators import singleton

# --- 导入 AUEmotionEngine ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .au_emotion_engine import AUEmotionEngine
# -------------------------

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MicroEmotionEngine")

# 微表情类别映射 - 基于CASME II数据集 (保持7类用于最终输出)
MICRO_EMOTION_CLASSES = {
    0: EmotionType.HAPPINESS,    # 高兴
    1: EmotionType.DISGUST,      # 厌恶
    2: EmotionType.SURPRISE,     # 惊讶
    3: EmotionType.FEAR,         # 恐惧 (模型不直接预测)
    4: EmotionType.SADNESS,      # 悲伤 (模型不直接预测)
    5: EmotionType.CONFUSION,    # 困惑/压抑 (映射自模型输出)
    6: EmotionType.NEUTRAL       # 中性（映射自模型输出 或 作为默认/低置信度结果）
}
# 基于CASMEⅡ数据集的类别映射关系 - 更正以匹配训练配置
LSTM_IDX_TO_EMOTION_TYPE = {
    0: EmotionType.HAPPINESS,  # 高兴 (训练索引 0)
    1: EmotionType.DISGUST,    # 厌恶 (训练索引 1)
    2: EmotionType.CONFUSION,  # 困惑/压抑 (训练索引 2)
    3: EmotionType.SURPRISE,   # 惊讶 (训练索引 3)
    4: EmotionType.NEUTRAL,    # 中性 (训练索引 4)
}
NUM_LSTM_CLASSES = 5 # LSTM 模型输出 5 类

# 数据归一化参数 (如果预处理需要)
# MEAN = [0.485, 0.456, 0.406] # 可能不需要，取决于CNN训练时的预处理
# STD = [0.229, 0.224, 0.225]

# --- START: Copied Model Definitions from 18_2/models.py ---

# --- Attention Block (Optional, mimics original project) ---
class AttentionBlock(nn.Module):
    def __init__(self, feature_dim, attention_hidden_dim=None):
        """
        Simple Attention Mechanism for sequence data.
        Args:
            feature_dim (int): Dimension of features at each time step (F).
            attention_hidden_dim (int, optional): Dimension of the hidden attention layer. Defaults to feature_dim // 2.
        """
        super(AttentionBlock, self).__init__()
        if attention_hidden_dim is None:
            attention_hidden_dim = feature_dim // 2 if feature_dim // 2 > 0 else 1 # Ensure positive dimension

        # Layers to compute attention scores
        self.attention_layer1 = nn.Linear(feature_dim, attention_hidden_dim)
        self.tanh = nn.Tanh()
        self.attention_layer2 = nn.Linear(attention_hidden_dim, 1) # Output one score per time step

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, F).
        Returns:
            torch.Tensor: Weighted sequence (B, T, F).
        """
        # Input shape: (B, T, F)
        attn_hidden = self.tanh(self.attention_layer1(x)) # (B, T, attention_hidden_dim)
        attn_logits = self.attention_layer2(attn_hidden)   # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # Softmax over the time dimension (T) -> (B, T, 1)
        weighted_sequence = x * attn_weights
        return weighted_sequence

# --- Stage 1 Model: CNN for Single Frame Feature Extraction ---
# Note: Removed the dependency on utils.NUM_CLASSES for portability
class CNNModel(nn.Module):
    def __init__(self, num_classes=7, input_channels=1): # Default num_classes to 7 as trained
        """
        Replicates the Keras CNN structure for feature extraction.
        Input is expected to be (B, C, H, W), e.g., (B, 1, 128, 128)
        """
        super(CNNModel, self).__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding='same') # Original used 5x5
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten and Dense layers
        self.flatten = nn.Flatten()
        # Calculate flattened size dynamically (assuming 128x128 input)
        # After 4 pooling layers (2x2 stride), 128 -> 64 -> 32 -> 16 -> 8
        # Make sure input image size matches this expectation (128x128)
        try:
            # Assuming 128x128 input leads to 8x8 feature map after pools
            flattened_size = 512 * 8 * 8
        except Exception as e:
             logger.error(f"Error calculating flattened size, assuming 128x128 input: {e}")
             flattened_size = 512 * 8 * 8 # Fallback, might be incorrect if input size changes

        self.fc1 = nn.Linear(flattened_size, 256) # Feature layer
        self.relu_fc1 = nn.ReLU()
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes) # Final classifier (not used if return_features=True)

    def forward(self, x, return_features=False):
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).
            return_features (bool): If True, return features before the final classifier.
        Returns:
            torch.Tensor: Output logits (B, num_classes) or features (B, feature_dim=256).
        """
        # Input shape B, C, H, W
        if x.shape[2] != 128 or x.shape[3] != 128:
             logger.warning(f"CNNModel received input with unexpected spatial dimensions: {x.shape}. Expected (B, C, 128, 128).")
             # Potential resizing needed here or in preprocessing. Assuming preprocessing handles it.
             
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.drop3(self.relu3(self.conv3(x))))
        x = self.pool4(self.drop4(self.relu4(self.conv4(x))))

        x = self.flatten(x)
        # Check flattened size consistency (optional debug)
        # if hasattr(self, 'fc1') and x.shape[1] != self.fc1.in_features:
        #     logger.error(f"Flattened size mismatch! Expected {self.fc1.in_features}, got {x.shape[1]}")
        #     # Handle error or proceed with caution

        features = self.relu_fc1(self.fc1(x)) # Features after first FC layer, shape (B, 256)

        if return_features:
            return features

        # Only execute if not returning features
        x_out = self.drop_fc1(features)
        x_out = self.fc2(x_out) # Logits
        return x_out

# --- Stage 2 Model: LSTM Only (using pre-extracted features) ---
class LSTMOnlyModel(nn.Module):
    def __init__(self,
                 input_feature_dim, # Dimension of features from CNN
                 lstm_hidden_size=256,
                 lstm_num_layers=1,
                 num_classes=5, # Trained on 5 classes
                 sequence_length=64, # Expected fixed sequence length
                 use_attention=False, # Default based on train_lstm.py config
                 dropout_lstm=0.3,
                 dropout_fc=0.5):
        """
        LSTM model for sequence classification using pre-extracted features.
        Input shape: (B, T, input_feature_dim)
        """
        super(LSTMOnlyModel, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.use_attention = use_attention

        # 1. LSTM Layers
        self.lstm1 = nn.LSTM(input_size=input_feature_dim,
                             hidden_size=lstm_hidden_size,
                             num_layers=lstm_num_layers,
                             batch_first=True,
                             bidirectional=False)

        # 2. Optional Attention Layer
        if self.use_attention:
            self.attention = AttentionBlock(feature_dim=lstm_hidden_size)

        # 3. Second LSTM Layer
        lstm2_input_size = lstm_hidden_size
        self.lstm2 = nn.LSTM(input_size=lstm2_input_size,
                             hidden_size=lstm_hidden_size // 2,
                             num_layers=lstm_num_layers,
                             batch_first=True,
                             bidirectional=False)

        self.dropout_lstm_out = nn.Dropout(dropout_lstm)

        # 4. Final Dense Layers
        fc1_input_size = lstm_hidden_size // 2
        self.fc1 = nn.Linear(fc1_input_size, fc1_input_size // 2)
        self.dropout_fc1 = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(fc1_input_size // 2, num_classes) # Output 5 classes
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature sequence tensor (B, T, input_feature_dim).
                               T should be self.sequence_length.
        Returns:
            torch.Tensor: Output logits (B, num_classes=5).
        """
        # Input shape: (B, T, input_feature_dim)
        if x.shape[1] != self.sequence_length:
             logger.warning(f"LSTMOnlyModel received sequence length {x.shape[1]}, expected {self.sequence_length}. Padding/truncation should happen before model.")
             # Assuming input 'x' is already padded/truncated to self.sequence_length

        # 1. Pass through first LSTM
        lstm1_out, _ = self.lstm1(x) # Shape (B, T, lstm_hidden_size)

        # 2. Optional Attention
        if self.use_attention:
             lstm1_out = self.attention(lstm1_out) # Shape (B, T, lstm_hidden_size)

        # 3. Pass through second LSTM
        lstm2_out, _ = self.lstm2(lstm1_out) # Shape (B, T, lstm_hidden_size // 2)

        # --- Use output of the last time step ---
        lstm_last_step_out = lstm2_out[:, -1, :] # Shape (B, lstm_hidden_size // 2)

        lstm_final_out = self.dropout_lstm_out(lstm_last_step_out)

        # 4. Dense Layers
        dense_hidden = self.fc1(lstm_final_out) # Shape (B, fc1_input_size // 2)
        dense_out = self.dropout_fc1(dense_hidden)
        logits = self.fc2(dense_out) # Shape (B, num_classes=5)

        return logits

# --- END: Copied Model Definitions ---


# Remove or comment out the old model definition
# class AdvancedMicroExpressionModel(nn.Module):
#    ... (old code) ...

@singleton
class MicroEmotionEngine:
    """微表情分析引擎，负责分析面部微表情"""
    
    _instance = None  # 单例实例
    _initialized = False # 添加类级别的 _initialized 标志
    
    def __init__(self,
                 config: ConfigManager, # 新的类型提示
                 event_bus: EventBus,
                 au_engine: Optional['AUEngine'] = None # 重命名为 au_engine
                 ): # 移除 *args, **kwargs
        """
        初始化微表情分析引擎
        
        Args:
            config: 配置管理器实例
            event_bus: 事件总线实例
            au_engine: AU引擎实例 (可选)
        """
        if MicroEmotionEngine._initialized: # 检查类级别的标志
            return
            
        logger.info("初始化微表情分析引擎...")

        self.config_manager = config
        self.event_bus = event_bus
        self.au_engine = au_engine # 存储为 self.au_engine
        if self.au_engine:
             logger.info("[MicroEmotionEngine] AUEngine instance provided and linked.")
        else:
             logger.warning("[MicroEmotionEngine] AUEngine instance not provided. AU assistance disabled.")

        self.config = self.config_manager.get('micro', {})
        self.system_config = self.config_manager.get('system', {})
        
        # 线程管理
        self.active_threads = set()
        self.thread_lock = threading.Lock()
        self.max_concurrent_threads = 3
        
        # 特征缓冲区设置
        # self.feature_buffer_size = 32  # REMOVED - Use lstm_sequence_length directly
        self.feature_buffers = {}      # 存储每个face_id的特征序列
        self._face_last_update = {}    # 记录每个face_id最后更新时间
        self.last_processed_frame_id = {}  # 记录每个face_id最后处理的帧ID
        self.prediction_buffer = {}    # 存储每个face_id的最近预测
        self.buffer_size = 5          # 保留最近5帧的预测
        
        # 预测设置
        self.prob_diff_threshold = 0.10  # 降低阈值，使其更容易检测到表情变化
        
        # 类别权重 (重置为训练配置的 repeats_per_label)
        self.class_weights = {
            0: 2.0,    # Happiness (训练 repeats)
            1: 2.0,    # Disgust (训练 repeats)
            2: 15.0,   # Confusion (训练 repeats for Repression)
            3: 3.0,    # Surprise (训练 repeats)
            4: 1.0     # Neutral (训练 repeats for Others)
        }
        
        # 类别阈值 (保持之前的调整，后续可再优化)
        self.class_thresholds = {
            0: 0.20,   # happiness - 提高阈值
            1: 0.15,   # disgust 
            2: 0.15,   # confusion
            3: 0.15,   # surprise
            4: 0.30    # neutral (保持较高以避免误判)
        }
        
        # LSTM模型配置
        self.lstm_config = {
            'input_feature_dim': 256,
            'lstm_hidden_size': 256,
            'lstm_num_layers': 1,
            'dropout_lstm': 0.627151707302907,
            'dropout_fc': 0.3745888760454624,
            'use_attention': False
        }
        
        # 初始化其他属性
        self.cnn_model = None
        self.lstm_models = None
        self.current_result = None
        self.enabled = True
        self.running = False
        self.paused = False
        
        # 获取模型路径和配置
        self.models_dir = self.config_manager.get("system.models_dir", "enhance_hidden/models")
        self.cnn_model_path = self.config_manager.get("micro.cnn_model_path", os.path.join(self.models_dir, "casme2_cnn_stage1_balanced_cnn_best.pth"))
        self.lstm_models_dir = self.config_manager.get("micro.lstm_models_dir", os.path.join(self.models_dir, "lstm_ensemble_models"))

        # 模型参数 (CNN params same, LSTM params define the architecture to load)
        self.input_size = tuple(self.config_manager.get("micro.cnn_input_size", [128, 128]))
        self.cnn_input_channels = self.config_manager.get("micro.cnn_input_channels", 1)
        self.cnn_feature_dim = self.config_manager.get("micro.cnn_feature_dim", 256)
        self.lstm_sequence_length = self.config_manager.get("micro.lstm_sequence_length", 64)
        self.lstm_hidden_size = self.config_manager.get("micro.lstm_hidden_size", 128)
        self.lstm_num_layers = self.config_manager.get("micro.lstm_num_layers", 1)
        self.lstm_use_attention = self.config_manager.get("micro.lstm_use_attention", False)
        self.lstm_dropout = self.config_manager.get("micro.lstm_dropout", 0.3)
        self.fc_dropout = self.config_manager.get("micro.fc_dropout", 0.5)

        self.detection_interval = self.config_manager.get("micro.detection_interval", 10)
        self.threshold = self.config_manager.get("micro.threshold", 0.6)
        self.use_gpu = self.config_manager.get("micro.gpu_acceleration", True) and torch.cuda.is_available()
        self.enabled = self.config_manager.get("micro.enabled", True)
        
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        
        # --- Model Placeholders ---
        self.cnn_model: Optional[CNNModel] = None
        # Changed to store a list of LSTM models
        self.lstm_models: List[LSTMOnlyModel] = []
        # --- End Model Placeholders ---
        
        # 状态变量
        self.is_running = False
        self.is_paused = False
        self.result_lock = threading.Lock()
        self.current_result: Optional[EmotionResult] = None
        self.processing_time = 0.0 # Will now reflect avg LSTM time maybe?
        
        # --- Feature Buffer (replaces frame buffer) ---
        # Stores extracted CNN features for each face
        # Key: face_id, Value: deque of feature tensors (F,)
        self.feature_buffers = {}
        self.last_processed_frame_id = {} # Keep this to control processing frequency

        # 光流计算器 (保留，但在此版本中可能不直接使用)
        self.use_optical_flow = self.config_manager.get("micro.use_optical_flow", False) # Defaulting to False for CNN-LSTM
        self.prev_gray = None
        self.optical_flow_buffer = {}
        
        self.config_manager.register_change_callback("micro", self._on_config_changed)
        self.config_manager.register_change_callback("system", self._on_system_config_changed)
        self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        
        # 加载模型
        self._load_models()
        
        MicroEmotionEngine._initialized = True
        logger.info(f"微表情分析引擎 (CNN-LSTM Ensemble) 初始化完成，使用设备: {self.device}")
    
    def _on_config_changed(self, path: str, value: Any):
        """配置变更回调"""
        reload_models = False
        reset_buffers = False

        if path == "micro.threshold":
            logger.info(f"微表情识别阈值已更新为: {value}")
            self.threshold = value
        elif path == "micro.cnn_model_path":
             logger.info(f"CNN模型路径已更新: {value}")
             self.cnn_model_path = value
             reload_models = True
        # --- Handle LSTM Models Directory Path --- 
        elif path == "micro.lstm_models_dir":
             logger.info(f"LSTM模型目录已更新: {value}")
             self.lstm_models_dir = value
             reload_models = True # Reload all LSTM models
        # --- End Handle LSTM Models Directory --- 
        elif path == "micro.cnn_input_size":
             logger.info(f"CNN输入尺寸已更新: {value}")
             self.input_size = tuple(value)
             reset_buffers = True
             reload_models = True
        elif path == "micro.lstm_sequence_length":
             logger.info(f"LSTM序列长度已更新: {value}")
             self.lstm_sequence_length = value
             reset_buffers = True
        elif path == "micro.lstm_hidden_size" or path == "micro.lstm_num_layers" or \
             path == "micro.lstm_use_attention" or path == "micro.lstm_dropout" or path == "micro.fc_dropout":
             logger.info(f"LSTM架构参数 '{path}' 已更新: {value}")
             # Update the attribute used when creating model instances
             # This assumes the attribute name matches the config key suffix
             attr_name = path.split('.')[-1]
             if hasattr(self, attr_name):
                 setattr(self, attr_name, value)
             else:
                 logger.warning(f"Attempted to update non-existent attribute {attr_name} from config {path}")
             reload_models = True # LSTM structure changed, reload all
        elif path == "micro.detection_interval":
            logger.info(f"检测间隔已更新为: {value}")
            self.detection_interval = value
        elif path == "micro.gpu_acceleration":
            logger.debug(f"[_on_config_changed] Received config change: micro.gpu_acceleration = {value}")
            if value != self.use_gpu:
                logger.info(f"GPU加速已{'启用' if value else '禁用'}")
                self.use_gpu = value and torch.cuda.is_available()
                new_device = torch.device("cuda:0" if self.use_gpu else "cpu")
                if new_device != self.device:
                    self.device = new_device
                    logger.info(f"设备已更改为: {self.device}，重新加载模型...")
                    reload_models = True
        elif path == "micro.use_optical_flow":
            logger.info(f"光流分析已{'启用' if value else '禁用'}")
            self.use_optical_flow = value
            reset_buffers = True
        elif path == "micro.enabled":
             logger.info(f"微表情引擎已{'启用' if value else '禁用'}")
             self.enabled = value
             if not value:
                 self.clear_buffers()

        if reload_models:
            self._load_models() # Reload both CNN and all LSTM models
        if reset_buffers:
            self.clear_buffers()
    
    def _on_system_config_changed(self, path: str, value: Any):
        """系统配置变更回调"""
        if path == "system.models_dir":
            logger.info(f"模型目录已更新为: {value}")
            old_models_dir = self.models_dir
            self.models_dir = value
            reload_needed = False
            # Update CNN path if it was using the old default
            old_cnn_default = os.path.join(old_models_dir, os.path.basename(self.config_manager.get_default("micro.cnn_model_path", "")))
            if self.cnn_model_path == old_cnn_default:
                self.cnn_model_path = os.path.join(self.models_dir, os.path.basename(self.cnn_model_path))
                logger.info(f"CNN模型路径自动更新为: {self.cnn_model_path}")
                reload_needed = True
            # --- Update LSTM Directory Path --- 
            old_lstm_default_dir = os.path.join(old_models_dir, os.path.basename(self.config_manager.get_default("micro.lstm_models_dir", "")))
            if self.lstm_models_dir == old_lstm_default_dir:
                 self.lstm_models_dir = os.path.join(self.models_dir, os.path.basename(self.lstm_models_dir))
                 logger.info(f"LSTM模型目录自动更新为: {self.lstm_models_dir}")
                 reload_needed = True
            # --- End Update --- 

            if reload_needed:
                self._load_models()
    
    def _on_face_detected(self, event: Event):
        """人脸检测事件处理 - 修改为处理单帧并提取特征"""
        logger.debug(f"[_on_face_detected] Received FACE_DETECTED event. Enabled: {self.enabled}, Running: {self.is_running}, Paused: {self.is_paused}")
        if not self.enabled or not self.is_running or self.is_paused or self.cnn_model is None:
            return
            
        # 检查当前活动线程数
        with self.thread_lock:
            if len(self.active_threads) >= self.max_concurrent_threads:
                logger.warning(f"[_on_face_detected] 达到最大并发线程数 ({self.max_concurrent_threads})，跳过此帧分析")
                return
            
        event_data = event.data
        if not event_data or "face" not in event_data or "frame" not in event_data:
            return
            
        face: FaceDetection = event_data["face"]
        frame: np.ndarray = event_data["frame"]
        frame_id: int = event_data.get("frame_id", 0)
        face_id: int = face.face_id

        # 1. 预处理当前帧 (for CNN)
        frame_tensor = self.preprocess_single_frame_cnn(frame, face)

        if frame_tensor is None:
            logger.debug(f"[_on_face_detected] Frame preprocessing failed for face {face_id}, frame {frame_id}.")
            return

        # 2. 使用 CNN 提取特征
        feature_vector = self._extract_cnn_features(frame_tensor)

        if feature_vector is None:
            logger.debug(f"[_on_face_detected] Feature extraction failed for face {face_id}, frame {frame_id}.")
            return

        # 3. 将特征添加到缓冲区
        self._add_feature_to_buffer(face_id, feature_vector, frame_id)

        # 4. 检查是否需要进行 LSTM 分类
        if face_id in self.feature_buffers and len(self.feature_buffers[face_id]) >= self.lstm_sequence_length:
            last_processed = self.last_processed_frame_id.get(face_id, -self.detection_interval -1)
            if frame_id - last_processed >= self.detection_interval:
                self.last_processed_frame_id[face_id] = frame_id
                # 创建并启动分析线程
                analysis_thread = threading.Thread(
                    target=self._analyze_lstm_sequence_threaded_wrapper,
                    args=(face_id, frame_id),
                    daemon=True
                )
                with self.thread_lock:
                    self.active_threads.add(analysis_thread)
                analysis_thread.start()

    def _analyze_lstm_sequence_threaded_wrapper(self, face_id: int, frame_id: int):
        """线程包装器，确保线程计数正确管理"""
        try:
            self._analyze_lstm_sequence_threaded(face_id, frame_id)
        finally:
            with self.thread_lock:
                # 查找并移除当前线程
                current_thread = threading.current_thread()
                self.active_threads.discard(current_thread)
                logger.debug(f"[_analyze_lstm_sequence_threaded_wrapper] Thread finished, active threads: {len(self.active_threads)}")

    # --- New function to add features ---
    def _add_feature_to_buffer(self, face_id: int, feature_vector: torch.Tensor, frame_id: int):
        """添加特征向量到缓冲区
        Args:
            face_id: 人脸ID
            feature_vector: CNN提取的特征向量
            frame_id: 帧ID
        """
        if face_id not in self.feature_buffers:
            # Use lstm_sequence_length for maxlen
            self.feature_buffers[face_id] = deque(maxlen=self.lstm_sequence_length)
            logger.debug(f"[_add_feature_to_buffer] Initialized feature buffer for new face_id: {face_id} with maxlen={self.lstm_sequence_length}")

        # 确保feature_vector是CPU张量并且已分离
        feature_vector_cpu = feature_vector.detach().cpu()
        self.feature_buffers[face_id].append(feature_vector_cpu)

        # 记录缓冲区大小
        buffer_size = len(self.feature_buffers[face_id])
        logger.debug(f"[_add_feature_to_buffer] Feature buffer size for face_id {face_id}: {buffer_size} / {self.lstm_sequence_length}")
        
        # 清理长时间未更新的人脸缓冲区
        current_time = time.time()
        if not hasattr(self, '_last_cleanup_time'):
            self._last_cleanup_time = current_time
            self._face_last_update = {}
        
        # 更新人脸最后活动时间
        self._face_last_update[face_id] = current_time
        
        # 每30秒执行一次清理
        if current_time - self._last_cleanup_time > 30:
            self._cleanup_old_buffers()
            self._last_cleanup_time = current_time

    def _cleanup_old_buffers(self):
        """清理长时间未更新的缓冲区"""
        current_time = time.time()
        timeout = 60  # 60秒无更新则清理
        
        # 找出需要清理的face_ids
        faces_to_remove = []
        for face_id in list(self.feature_buffers.keys()):
            last_update = self._face_last_update.get(face_id, 0)
            if current_time - last_update > timeout:
                faces_to_remove.append(face_id)
        
        # 执行清理
        for face_id in faces_to_remove:
            if face_id in self.feature_buffers:
                del self.feature_buffers[face_id]
            if face_id in self._face_last_update:
                del self._face_last_update[face_id]
            if face_id in self.last_processed_frame_id:
                del self.last_processed_frame_id[face_id]
            # 清理预测缓冲区
            self._cleanup_prediction_buffer(face_id)
            logger.debug(f"[_cleanup_old_buffers] Cleaned up buffers for inactive face_id: {face_id}")
            
        if faces_to_remove:
            logger.info(f"[_cleanup_old_buffers] Cleaned up {len(faces_to_remove)} inactive face buffers")
            
    def clear_buffers(self):
        """清空所有缓冲区"""
        self.feature_buffers = {}
        self.last_processed_frame_id = {}
        self.prediction_buffer = {}  # 清空预测缓冲区
        self.prev_gray = None
        self.optical_flow_buffer = {}
        logger.info("所有缓冲区已清空 (Feature buffers, prediction buffers, optical flow)")

    # --- New function for LSTM analysis ---
    def _analyze_lstm_sequence_threaded(self, face_id: int, frame_id: int):
        """在线程中执行LSTM序列分析 (Ensemble)"""
        logger.debug(f"[_analyze_lstm_sequence_threaded] ENSEMBLE THREAD STARTED for face_id: {face_id}, ending frame_id: {frame_id}")
        if self.lstm_models is None or not self.lstm_models:
            logger.warning("[_analyze_lstm_sequence_threaded] LSTM models list is empty or None.")
            return
            
        try:
            # --- 1. 获取特征序列 ---
            buffer = self.feature_buffers.get(face_id)
            if buffer is None:
                logger.warning(f"[_analyze_lstm_sequence_threaded] No feature buffer found for face_id {face_id}.")
                return
            feature_sequence_list = list(buffer)
            if len(feature_sequence_list) != self.lstm_sequence_length:
                logger.warning(f"[_analyze_lstm_sequence_threaded] Feature sequence for face {face_id} has incorrect length ({len(feature_sequence_list)}/{self.lstm_sequence_length}). Aborting.")
                return
                
            # --- 2. 组装 LSTM 输入张量 ---
            try:
                if not all(isinstance(t, torch.Tensor) for t in feature_sequence_list):
                    logger.error(f"[_analyze_lstm_sequence_threaded] Non-tensor object found in feature buffer for face {face_id}. Types: {[type(t) for t in feature_sequence_list]}")
                    return
                feature_sequence_tensor = torch.stack(feature_sequence_list, dim=0)
                if len(feature_sequence_tensor.shape) != 2 or feature_sequence_tensor.shape[0] != self.lstm_sequence_length or feature_sequence_tensor.shape[1] != self.cnn_feature_dim:
                    logger.error(f"[_analyze_lstm_sequence_threaded] Stacked feature tensor has unexpected shape: {feature_sequence_tensor.shape}. Expected: [{self.lstm_sequence_length}, {self.cnn_feature_dim}]")
                    return
                lstm_input_batch = feature_sequence_tensor.unsqueeze(0)
                lstm_input_batch = lstm_input_batch.to(self.device)
                logger.debug(f"[_analyze_lstm_sequence_threaded] LSTM input tensor shape: {lstm_input_batch.shape}, Device: {lstm_input_batch.device}")
            except Exception as stack_e:
                logger.error(f"[_analyze_lstm_sequence_threaded] Error stacking feature tensors for face {face_id}: {stack_e}", exc_info=True)
                return

            # --- 3. LSTM 推理 (ENSEMBLE) ---
            all_probabilities = []
            total_inference_time = 0.0
            start_time_ensemble = time.time()

            with torch.no_grad():
                for i, model in enumerate(self.lstm_models):
                    try:
                        start_time_single = time.time()
                        model.eval()
                        logits = model(lstm_input_batch)
                        single_inference_time = time.time() - start_time_single
                        total_inference_time += single_inference_time
                        logger.debug(f"Model {i+1}/{len(self.lstm_models)} inference time: {single_inference_time:.4f}s")

                        if logits is None or len(logits.shape) != 2 or logits.shape[0] != 1 or logits.shape[1] != NUM_LSTM_CLASSES:
                            logger.error(f"Model {i+1} produced unexpected logits shape: {logits.shape if logits is not None else 'None'}. Skipping this model.")
                            continue

                        probabilities = F.softmax(logits, dim=1)
                        all_probabilities.append(probabilities)

                    except Exception as inference_err:
                        logger.error(f"Error during inference with LSTM model {i+1}: {inference_err}", exc_info=True)

            if not all_probabilities:
                logger.error("No valid probabilities obtained from any LSTM model in the ensemble.")
                return

            avg_inference_time = total_inference_time / len(all_probabilities) if all_probabilities else 0
            self.processing_time = avg_inference_time
            logger.debug(f"Ensemble inference complete. Avg time per model: {avg_inference_time:.4f}s. Total valid models: {len(all_probabilities)}")

            # --- 4. 计算平均概率并处理结果 --- 
            stacked_probs = torch.stack(all_probabilities, dim=0)
            mean_probs = torch.mean(stacked_probs, dim=0)
            final_probabilities_5_class = mean_probs[0]

            # 获取每个类别的概率
            probabilities_per_class = {
                LSTM_IDX_TO_EMOTION_TYPE[i].name.lower(): final_probabilities_5_class[i].item()
                for i in range(NUM_LSTM_CLASSES)
            }
            
            # 记录原始概率分布
            logger.debug(f"Face {face_id} raw probabilities: {probabilities_per_class}")
            
            # 应用类别权重调整概率
            adjusted_probs = final_probabilities_5_class.clone()
            for i in range(NUM_LSTM_CLASSES):
                adjusted_probs[i] = adjusted_probs[i] * self.class_weights[i]
            
            # 重新归一化概率
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # 获取调整后的top-3预测
            top3_values, top3_indices = torch.topk(adjusted_probs, k=3)
            
            # 记录调整后的top-3预测结果
            top3_str = "\n            ".join([
                f"{i+1}st: {LSTM_IDX_TO_EMOTION_TYPE[idx.item()].name} ({top3_values[i]:.3f})"
                for i, idx in enumerate(top3_indices)
            ])
            logger.debug(f"\n            Face {face_id} adjusted top-3 predictions:\n            {top3_str}\n            ")
            
            # 检查是否需要考虑次高概率的预测
            top1_idx = top3_indices[0].item()
            top2_idx = top3_indices[1].item()
            prob_diff = top3_values[0] - top3_values[1]
            
            # 如果top1是中性且概率差小于阈值，考虑使用top2
            final_pred = None
            final_prob = 0.0
            
            if top1_idx == 4 and prob_diff < self.prob_diff_threshold:  # 4是中性类别的索引
                if top3_values[1] >= self.class_thresholds[top2_idx]:
                    final_pred = LSTM_IDX_TO_EMOTION_TYPE[top2_idx]
                    final_prob = top3_values[1].item()
                    logger.debug(f"Using second highest prediction due to small probability difference")
            
            # 如果没有选择次高概率，使用最高概率
            if final_pred is None:
                if top3_values[0] >= self.class_thresholds[top1_idx]:
                    final_pred = LSTM_IDX_TO_EMOTION_TYPE[top1_idx]
                    final_prob = top3_values[0].item()
                else:
                    # 如果最高概率低于阈值，返回None
                    logger.debug(f"Top prediction probability {top3_values[0]:.3f} below threshold {self.class_thresholds[top1_idx]}")
                    return None
            
            # 使用类别特定的阈值
            class_threshold = self.class_thresholds[top1_idx]
            
            if final_prob >= class_threshold:
                # 应用平滑处理
                smoothed_result = self._smooth_prediction(
                    face_id,
                    final_pred,
                    final_prob,
                    probabilities_per_class
                )
                
                logger.info(f"Micro-expression DETECTED (Ensemble) for face {face_id}: {smoothed_result.emotion_type.name} ({smoothed_result.probability:.3f}) [Class Threshold: {class_threshold:.3f}]")
                
                # 发布原始的微表情分析结果
                self.event_bus.publish(EventType.RAW_MICRO_EMOTION_ANALYZED, { # 修改事件类型
                    "face_id": face_id,
                    "frame_id": frame_id,
                    "result": smoothed_result, # 这是主引擎的原始（平滑后）结果
                    "raw_probabilities": probabilities_per_class,
                    "frame_data": {"image": None, "timestamp": time.time()} # 占位符，实际图像需从上层获取或传递
                    # TODO: Consider how to get the relevant `frame` image here if needed by MicroEmotionAUEngine
                    # Currently, _analyze_lstm_sequence_threaded does not have direct access to the full frame image 
                    # that corresponds to the end of the sequence. This might require a design change if the image is essential.
                })
                
                 # 更新当前结果
                with self.result_lock:
                    self.current_result = smoothed_result
            else:
                logger.debug(f"[_analyze_lstm_sequence_threaded] Confidence {final_prob:.4f} below class threshold {class_threshold:.4f} for {final_pred.name}. No event published.")

        except Exception as e:
            logger.error(f"[_analyze_lstm_sequence_threaded] ENSEMBLE分析线程错误 for face {face_id}: {e}", exc_info=True)
        finally:
            logger.debug(f"[_analyze_lstm_sequence_threaded] ENSEMBLE THREAD FINISHED for face_id: {face_id}, ending frame_id: {frame_id}")


    # --- New combined loading function ---
    def _load_models(self):
        """加载 CNN 和 LSTM 模型"""
        logger.info("Loading CNN and LSTM models...")
        cnn_loaded = self._load_cnn_model()
        # Call the new function to load all LSTM models
        lstm_loaded = self._load_all_lstm_models()

        if cnn_loaded:
             logger.info("CNN model loaded successfully.")
        else:
             logger.error("CNN model loading FAILED.")

        if lstm_loaded:
             logger.info(f"LSTM models loaded successfully ({len(self.lstm_models)} models found).")
        else:
             logger.error("LSTM models loading FAILED.")

        return cnn_loaded and lstm_loaded

    def _load_cnn_model(self) -> bool:
        """加载 CNN 特征提取模型"""
        logger.debug(f"[_load_cnn_model] Attempting to load CNN model from: {self.cnn_model_path}")
        try:
            if not os.path.exists(self.cnn_model_path):
                logger.error(f"CNN模型文件不存在: {self.cnn_model_path}")
                self.cnn_model = None
                return False
            
            # Instantiate CNNModel (num_classes=7 as trained, input_channels=1)
            # Ensure self.input_size is correct (e.g., 128x128) for architecture
            self.cnn_model = CNNModel(num_classes=7, input_channels=self.cnn_input_channels)
            
            logger.debug(f"Loading CNN state dict from {self.cnn_model_path} to device {self.device}")
            checkpoint = torch.load(self.cnn_model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            state_dict = checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']

            self.cnn_model.load_state_dict(state_dict)
            self.cnn_model = self.cnn_model.to(self.device)
            self.cnn_model.eval() # Set to evaluation mode

            # Freeze CNN parameters
            for param in self.cnn_model.parameters():
                param.requires_grad = False
            logger.info("CNN backbone frozen and set to eval mode.")

            # Simple test forward pass
            try:
                 dummy_input = torch.randn(1, self.cnn_input_channels, self.input_size[0], self.input_size[1]).to(self.device)
                 with torch.no_grad():
                      features = self.cnn_model(dummy_input, return_features=True)
                 logger.info(f"CNN model loaded. Test forward pass feature shape: {features.shape}") # Should be [1, cnn_feature_dim]
                 if features.shape[1] != self.cnn_feature_dim:
                      logger.warning(f"Loaded CNN model feature dimension ({features.shape[1]}) does not match configured dimension ({self.cnn_feature_dim}). Check model architecture or config.")
            except Exception as test_e:
                 logger.error(f"Error during CNN test forward pass after loading: {test_e}", exc_info=True)
                 # self.cnn_model = None # Optional: Invalidate model if test fails
                 # return False

            return True
            
        except Exception as e:
            logger.error(f"加载 CNN 模型失败: {e}", exc_info=True)
            self.cnn_model = None
            return False
    
    # --- New Function to Load All LSTM Models --- 
    def _load_all_lstm_models(self) -> bool:
        """从指定目录加载所有 K-Fold LSTM 模型"""
        logger.debug(f"[_load_all_lstm_models] Attempting to load LSTM models from directory: {self.lstm_models_dir}")
        self.lstm_models = [] # Clear existing models first

        try:
            if not os.path.isdir(self.lstm_models_dir):
                logger.error(f"LSTM 模型目录不存在或不是一个目录: {self.lstm_models_dir}")
                return False

            model_files_found = []
            # Find all LSTM model files
            for filename in os.listdir(self.lstm_models_dir):
                # 放宽匹配条件
                if ("lstm" in filename.lower() or "fold" in filename.lower()) and filename.endswith(".pth"):
                    model_files_found.append(os.path.join(self.lstm_models_dir, filename))
                    logger.debug(f"Found LSTM model file: {filename}")

            if not model_files_found:
                logger.error(f"在目录 {self.lstm_models_dir} 中没有找到LSTM模型文件。")
                return False

            logger.info(f"找到 {len(model_files_found)} 个 LSTM 模型文件，准备加载...")

            # Load each found model
            for model_path in sorted(model_files_found): # Sort for consistent order
                try:
                    logger.debug(f"Loading LSTM model: {model_path}")
                    # Create a new instance for each model
                    model_instance = LSTMOnlyModel(
                        input_feature_dim=self.cnn_feature_dim,
                        lstm_hidden_size=self.lstm_hidden_size,
                        lstm_num_layers=self.lstm_num_layers,
                        num_classes=NUM_LSTM_CLASSES,
                        sequence_length=self.lstm_sequence_length,
                        use_attention=self.lstm_use_attention,
                        dropout_lstm=self.lstm_dropout,
                        dropout_fc=self.fc_dropout
                    )

                    checkpoint = torch.load(model_path, map_location=self.device)
                    state_dict = checkpoint # Default assume state_dict is the checkpoint itself
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                         state_dict = checkpoint['state_dict']
                         # Adjust keys if necessary (e.g., remove prefix)
                         state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

                    model_instance.load_state_dict(state_dict)
                    model_instance = model_instance.to(self.device)
                    model_instance.eval() # Set to evaluation mode
                    self.lstm_models.append(model_instance)
                    logger.debug(f"Successfully loaded and added LSTM model: {model_path}")
                except Exception as load_err:
                    # Log error for specific model but continue loading others
                    logger.error(f"加载 LSTM 模型 {model_path} 失败: {load_err}", exc_info=True)

            # Check if at least one model was loaded successfully
            if not self.lstm_models:
                logger.error("所有 LSTM 模型都加载失败。")
                return False

            logger.info(f"成功加载 {len(self.lstm_models)} 个 LSTM 模型。")

            # Optional: Test forward pass for the first loaded LSTM model
            if self.lstm_models:
                try:
                     dummy_input = torch.randn(1, self.lstm_sequence_length, self.cnn_feature_dim).to(self.device)
                     with torch.no_grad():
                          logits = self.lstm_models[0](dummy_input) # Test only the first one
                     logger.info(f"LSTM model[0] test forward pass output shape: {logits.shape}")
                     if logits.shape[1] != NUM_LSTM_CLASSES:
                          logger.error(f"Loaded LSTM model[0] output dimension ({logits.shape[1]}) != expected classes ({NUM_LSTM_CLASSES}).")
                except Exception as test_e:
                     logger.error(f"Error during LSTM model[0] test forward pass: {test_e}")

            return True # Return True if at least one model loaded

        except Exception as e:
            logger.error(f"加载所有 LSTM 模型时发生意外错误: {e}", exc_info=True)
            self.lstm_models = [] # Ensure list is empty on error
            return False

    # --- Update Preprocessing for CNN ---
    def preprocess_single_frame_cnn(self, frame: np.ndarray, face: FaceDetection) -> Optional[torch.Tensor]:
        """对单帧图像中的人脸进行预处理，适配CNN输入 (B=1, C=1, H, W)"""
        logger.debug("[preprocess_single_frame_cnn] Starting preprocessing...")
        logger.debug(f"[preprocess_single_frame_cnn] Input frame type: {type(frame)}, shape: {frame.shape if isinstance(frame, np.ndarray) else 'Not an ndarray'}")
        logger.debug(f"[preprocess_single_frame_cnn] Input face_chip type: {type(face.face_chip)}, shape: {face.face_chip.shape if isinstance(face.face_chip, np.ndarray) else 'Not an ndarray or None'}")

        try:
            # 1. Crop face (using face_chip if available, else crop from frame with border)
            if face.face_chip is not None and face.face_chip.size > 0:
                logger.debug(f"[preprocess_single_frame_cnn] Attempting to use face_chip. Type: {type(face.face_chip)}, Size: {face.face_chip.size}")
                face_img = face.face_chip
                logger.debug("[preprocess_single_frame_cnn] Using face_chip.")
            elif face.face_box:
                logger.debug(f"[preprocess_single_frame_cnn] face_chip is None or empty, attempting to crop from frame using face_box: {face.face_box}")
                x1, y1, x2, y2 = map(int, face.face_box.to_tlbr())
                # Add border (e.g., 10% of width/height)
                border_x = int((x2 - x1) * 0.1)
                border_y = int((y2 - y1) * 0.1)
                crop_x1 = max(0, x1 - border_x)
                crop_y1 = max(0, y1 - border_y)
                crop_x2 = min(frame.shape[1], x2 + border_x)
                crop_y2 = min(frame.shape[0], y2 + border_y)
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    face_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    logger.debug(f"[preprocess_single_frame_cnn] Cropped from frame: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
                else:
                     logger.warning("[preprocess_single_frame_cnn] Invalid crop coordinates after adding border.")
                     return None
            else:
                logger.warning("[preprocess_single_frame_cnn] No face_chip or face_box available.")
                return None

            if face_img is None or face_img.size == 0:
                 logger.warning("[preprocess_single_frame_cnn] Face image is empty after cropping/selection.")
                 return None

            # 2. Convert to Grayscale (ensure input channel C=1)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            elif len(face_img.shape) == 2:
                gray_face = face_img # Already grayscale
            else:
                 logger.warning(f"[preprocess_single_frame_cnn] Unhandled face image shape: {face_img.shape}")
                 return None

            # 3. Resize to CNN input size (e.g., 128x128)
            # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging generally good
            interpolation = cv2.INTER_AREA if gray_face.shape[0] > self.input_size[0] else cv2.INTER_LINEAR
            resized_face = cv2.resize(gray_face, self.input_size, interpolation=interpolation) # H, W

            # 4. Normalize pixel values to [0, 1] (standard practice before ToTensor)
            normalized_face = resized_face.astype(np.float32) / 255.0

            # 5. Convert to PyTorch Tensor and add Channel dimension (H, W) -> (C=1, H, W)
            # The CNN model expects (B, C, H, W), so we need [1, H, W] here.
            tensor_face = torch.from_numpy(normalized_face).unsqueeze(0) # Shape: [1, H, W]

            # 6. Add Batch dimension -> (B=1, C=1, H, W)
            batch_tensor = tensor_face.unsqueeze(0) # Shape: [1, 1, H, W]

            logger.debug(f"[preprocess_single_frame_cnn] Preprocessed tensor shape: {batch_tensor.shape}")
            # Note: No further normalization (like ImageNet mean/std) is applied here,
            # assuming the CNN was trained on [0, 1] grayscale inputs. If it was trained
            # with specific normalization, that should be added here.
            return batch_tensor # Return tensor ready for CNN input [1, 1, H, W]
            
        except Exception as e:
            logger.error(f"[preprocess_single_frame_cnn] Error: {e}", exc_info=True) # Ensure exc_info is True for full traceback
            return None
    
    # --- Implement Feature extraction using CNN ---
    def _extract_cnn_features(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """ 使用加载的CNN模型提取特征 (Input: B,C,H,W tensor, B=1) """
        if self.cnn_model is None:
            logger.warning("[_extract_cnn_features] CNN model not loaded.")
            return None
            
        # Ensure input tensor has the correct shape (B=1, C=1, H, W)
        if frame_tensor is None or len(frame_tensor.shape) != 4 or frame_tensor.shape[0] != 1 or frame_tensor.shape[1] != self.cnn_input_channels:
             logger.error(f"[_extract_cnn_features] Invalid input tensor shape: {frame_tensor.shape if frame_tensor is not None else 'None'}. Expected [1, {self.cnn_input_channels}, H, W]")
             return None

        try:
            # Ensure tensor is on the correct device
            frame_tensor_device = frame_tensor.to(self.device)

            self.cnn_model.eval() # Ensure eval mode
            with torch.no_grad():
                # Get features, shape should be (B=1, feature_dim)
                features = self.cnn_model(frame_tensor_device, return_features=True)

            # Validate output shape
            if features is None or len(features.shape) != 2 or features.shape[0] != 1:
                 logger.error(f"[_extract_cnn_features] Unexpected feature tensor shape from CNN: {features.shape if features is not None else 'None'}. Expected [1, feature_dim]")
                 return None

            # Remove batch dimension -> (feature_dim,)
            feature_vector = features[0]
            logger.debug(f"[_extract_cnn_features] Extracted feature vector shape: {feature_vector.shape}") # Should be [cnn_feature_dim]

            # Return the feature vector (potentially still on GPU, handled in _add_feature_to_buffer)
            return feature_vector

        except Exception as e:
            logger.error(f"[_extract_cnn_features] Error during CNN feature extraction: {e}", exc_info=True)
            return None # Ensure correct indentation


    # --- Analysis needs update ---
    # Remove the commented-out old function definitions entirely
    # --- End Analysis update needed ---

    
    def start(self):
        """启动微表情分析引擎"""
        if self.is_running:
            logger.warning("微表情分析引擎已经在运行中")
            return
        # Check if CNN and *at least one* LSTM model are loaded
        if self.cnn_model is None or not self.lstm_models:
             logger.error("CNN model or LSTM models not loaded. Cannot start engine. Trying to load again...")
             if not self._load_models():
                  logger.error("Failed to load models. Engine cannot start.")
                  return
             # Re-check after attempting load
             if self.cnn_model is None or not self.lstm_models:
                  logger.error("Still failed to load models after retry. Engine cannot start.")
             return
        
        self.is_running = True
        self.is_paused = False
        logger.info(f"微表情分析引擎 (CNN-LSTM Ensemble, {len(self.lstm_models)} LSTM models) 已启动") # Log count
        self.event_bus.publish(EventType.ENGINE_STARTED, {"engine": "micro"})
    
    def stop(self):
        """停止微表情分析引擎"""
        if not self.is_running:
            logger.warning("微表情分析引擎未运行")
            return
        
        self.is_running = False
        # 等待所有活动线程完成
        with self.thread_lock:
            active_threads = list(self.active_threads)
        for thread in active_threads:
            thread.join(timeout=1.0)  # 设置超时以防止永久等待
        
        self.clear_buffers()
        logger.info("微表情分析引擎 (CNN-LSTM Ensemble) 已停止")
        self.event_bus.publish(EventType.ENGINE_STOPPED, {"engine": "micro"})
    
    def pause(self):
        """暂停微表情分析引擎"""
        if not self.is_running:
            logger.warning("微表情分析引擎未运行")
            return
        self.is_paused = True
        logger.info("微表情分析引擎 (CNN-LSTM Ensemble) 已暂停")
        self.event_bus.publish(EventType.ENGINE_PAUSED, {"engine": "micro"})
    
    def resume(self):
        """恢复微表情分析引擎"""
        if not self.is_running:
            logger.warning("微表情分析引擎未运行")
            return
        if self.is_paused:
            self.is_paused = False
            logger.info("微表情分析引擎 (CNN-LSTM Ensemble) 已恢复")
            self.event_bus.publish(EventType.ENGINE_RESUMED, {"engine": "micro"})
        else:
            logger.info("微表情分析引擎 (CNN-LSTM Ensemble) 已在运行中，无需恢复。")
    
    def get_current_result(self) -> Optional[EmotionResult]:
        """获取当前微表情分析结果 (7类)"""
        with self.result_lock:
            return self.current_result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计数据 (更新缓冲区信息)"""
        total_features = sum(len(buffer) for buffer in self.feature_buffers.values())
        # Note: processing_time might need recalculation based on CNN+LSTM steps
        return {
            "processing_time_last_lstm": self.processing_time, # Reflects last LSTM step time
            "fps_lstm_approx": 1.0 / self.processing_time if self.processing_time > 0 else 0.0,
            "cached_sequences": len(self.feature_buffers),
            "total_cached_features": total_features
        }

    # --- Drawing result remains the same, uses self.current_result (7-class) ---
    def draw_result(self, image: np.ndarray, face_detection: Optional[FaceDetection] = None) -> np.ndarray:
        """在图像上绘制微表情分析结果 (使用7类结果)"""
        if image is None: return image
        emotion_result = self.get_current_result()
        if emotion_result is None: return image
        
        vis_img = image.copy()
        if face_detection and face_detection.face_box:
            x1, y1, x2, y2 = map(int, face_detection.face_box.to_tlbr())
            # Display the mapped 7-class emotion
            emotion_text = f"Micro: {emotion_result.emotion_type.value} ({emotion_result.probability:.2f})"
            # Position text above the box
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(vis_img, emotion_text, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2) # Changed color slightly
        return vis_img

    # --- Old preprocessing function, comment out or remove ---
    # def preprocess_single_frame(self, frame: np.ndarray, face: FaceDetection) -> Optional[torch.Tensor]:
    #    """(OLD) 对单帧图像中的人脸进行预处理"""
    #    # ... (Keep commented out or remove) ...
    #    pass

    def _smooth_prediction(self, face_id: int, current_pred: EmotionType, 
                         current_prob: float, probabilities: Dict[str, float]) -> EmotionResult:
        """平滑预测结果
        Args:
            face_id: 人脸ID
            current_pred: 当前预测的情绪类型
            current_prob: 当前预测的概率
            probabilities: 所有类别的概率分布
        Returns:
            EmotionResult: 平滑后的预测结果
        """
        if face_id not in self.prediction_buffer:
            self.prediction_buffer[face_id] = []
            
        buffer = self.prediction_buffer[face_id]
        buffer.append((current_pred, current_prob, probabilities))
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
            
        # 如果缓冲区太小，直接返回当前预测
        if len(buffer) < 3:
            return EmotionResult(
                emotion_type=current_pred,
                probability=current_prob
            )
            
        # 计算每个类别的加权概率和出现次数
        emotion_probs = defaultdict(float)
        emotion_counts = defaultdict(int)
        weights = np.linspace(0.6, 1.0, len(buffer))  # 最近的预测权重更大
        
        # 创建反向映射字典
        emotion_to_idx = {emotion_type: idx for idx, emotion_type in LSTM_IDX_TO_EMOTION_TYPE.items()}
        
        # 首先统计每个情绪类型的出现次数和加权概率
        for (emotion, prob, probs), weight in zip(buffer, weights):
            emotion_counts[emotion] += 1
            emotion_probs[emotion] += prob * weight
            
            # 同时考虑概率分布中的次高概率
            for e_type_str, p in probs.items():
                try:
                    e_type = EmotionType[e_type_str.upper()]
                    if e_type != emotion:
                        # 使用反向映射获取索引
                        idx = emotion_to_idx.get(e_type)
                        if idx is not None and p > self.class_thresholds.get(idx, 0.5):
                            emotion_counts[e_type] += 0.5  # 给予较小的权重
                            emotion_probs[e_type] += p * weight * 0.5
                except (KeyError, ValueError):
                    continue
        
        # 计算平均概率和稳定性分数
        stability_scores = {}
        for emotion in emotion_probs:
            # 计算平均概率
            avg_prob = emotion_probs[emotion] / sum(weights)
            # 计算稳定性分数 (考虑出现次数和概率)
            stability = (emotion_counts[emotion] / len(buffer)) * avg_prob
            stability_scores[emotion] = stability
        
        # 选择最稳定的情绪类型
        max_stability = 0
        smoothed_emotion = current_pred
        smoothed_prob = current_prob
        
        for emotion, stability in stability_scores.items():
            if stability > max_stability:
                max_stability = stability
                smoothed_emotion = emotion
                smoothed_prob = emotion_probs[emotion] / sum(weights)
        
        # 如果最稳定的情绪不是当前预测，且差异显著，记录变化
        if smoothed_emotion != current_pred and abs(smoothed_prob - current_prob) > 0.1:
            logger.info(f"Smoothing changed prediction for face {face_id}: {current_pred.name}({current_prob:.3f}) -> {smoothed_emotion.name}({smoothed_prob:.3f})")
        
        return EmotionResult(
            emotion_type=smoothed_emotion,
            probability=smoothed_prob
        )

    def _cleanup_prediction_buffer(self, face_id: int):
        """清理指定face_id的预测缓冲区"""
        if face_id in self.prediction_buffer:
            del self.prediction_buffer[face_id]
            logger.debug(f"已清理face_id {face_id}的预测缓冲区")