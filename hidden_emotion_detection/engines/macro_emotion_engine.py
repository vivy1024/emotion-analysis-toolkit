#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观表情分析引擎
基于EmotionResNet深度学习模型实现7类基本表情识别
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
from collections import OrderedDict
import torchvision.models as models
import torchvision.transforms as transforms

from hidden_emotion_detection.config import ConfigManager, config_manager as global_config_manager # 新的导入
from hidden_emotion_detection.core.data_types import EmotionResult, EmotionType, FaceBox, Event, EventType, FaceDetection
from hidden_emotion_detection.core.event_bus import EventBus

# --- 导入 AUEmotionEngine ---
from typing import TYPE_CHECKING # 用于类型提示循环依赖
if TYPE_CHECKING:
    from .au_emotion_engine import AUEmotionEngine 
# -------------------------

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MacroEmotionEngine")

# 表情类别映射
EMOTION_CLASSES = {
    0: EmotionType.ANGER,      # 愤怒
    1: EmotionType.DISGUST,    # 厌恶
    2: EmotionType.FEAR,       # 恐惧
    3: EmotionType.HAPPINESS,  # 高兴
    4: EmotionType.SADNESS,    # 悲伤
    5: EmotionType.SURPRISE,   # 惊讶
    6: EmotionType.NEUTRAL     # 中性
}

# FER2013数据集均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class EmotionResNet(nn.Module):
    """基于ResNet架构的表情识别模型"""
    
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()
        
        # 加载预训练的ResNet18模型
        # 使用 weights='DEFAULT' 获取推荐的预训练权重
        self.resnet = models.resnet18(weights='DEFAULT') 
        
        # 修改第一个卷积层以接受灰度图像 (1 通道输入)
        # ResNet18 的 conv1 原本是 Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 我们保留 64 个输出通道，但输入通道改为 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后一个全连接层以输出指定数量的情绪类别
        num_features = self.resnet.fc.in_features # 获取 ResNet18 fc 层的输入特征数 (通常是 512)
        # --- 恢复与训练脚本一致的 fc 层结构 --- 
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512), # 对应 fc.0
            nn.ReLU(),                   # 对应 fc.1
            nn.Dropout(0.5),             # 对应 fc.2
            nn.Linear(512, num_classes)  # 对应 fc.3
        )
        # -----------------------------------------
    
    def forward(self, x):
        # 直接调用修改后的 resnet 的 forward 方法
        return self.resnet(x)

class MacroEmotionEngine:
    """宏观表情分析引擎，进行基本表情识别"""
    
    _instance = None  # 单例实例
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(MacroEmotionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self,
                 config: ConfigManager, # 新的类型提示
                 event_bus: EventBus,
                 au_engine: Optional['AUEngine'] = None # 重命名为 au_engine
                 ):
        """
        初始化宏观表情分析引擎
        
        Args:
            config: 配置管理器实例
            event_bus: 事件总线实例
            au_engine: AU引擎实例 (可选)
        """
        if getattr(self, '_initialized', False):
            return
            
        logger.info("初始化宏观表情分析引擎...")
        
        self.config_manager = config
        self.event_bus = event_bus
        self.au_engine = au_engine # 存储为 self.au_engine
        if self.au_engine:
             logger.info("[MacroEmotionEngine] AUEngine instance provided and linked.")
        else:
             logger.warning("[MacroEmotionEngine] AUEngine instance not provided. AU assistance disabled.")
        
        self.config = self.config_manager.get('macro', {})
        self.system_config = self.config_manager.get('system', {})
        
        self.model_path = self.config.get('model_path')
        self.input_size = (self.config.get('image_size', 48), self.config.get('image_size', 48))
        self.device = torch.device(self.system_config.get('device', 'cpu'))
        self.batch_size = self.config.get('batch_size', 1)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)
        
        self.model: Optional[torch.nn.Module] = None
        self.transform = self._get_transform()
        
        self.is_running = False
        self.is_paused = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.last_analysis_time = 0
        self.analysis_interval = 1.0 / self.config.get('fps', 10)
        
        self.result_lock = threading.Lock()
        self.current_result = None
        self.processing_time = 0.0
        
        self.results_cache = {}
        self.max_cache_size = 50
        
        # 订阅配置变更
        self.config_manager.register_change_callback('macro', self._on_config_changed)
        self.config_manager.register_change_callback('system', self._on_system_config_changed)
        
        # 订阅人脸检测事件
        self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        
        self._load_model()
        
        self._initialized = True
        logger.info(f"宏观表情分析引擎初始化完成，使用设备: {self.device}")
    
    def _get_transform(self):
        image_size = self.config.get('image_size', 48)
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def _on_config_changed(self, key: str, value: Any):
        """配置变更回调"""
        logger.debug(f"[_on_config_changed] Received config change: {key} = {value}")
        self.config = self.config_manager.get('macro', {})
        if key == "model_path":
            logger.info(f"模型路径已更新为: {value}")
            self.model_path = value
            self._load_model()
        elif key == "fps":
            logger.info(f"分析间隔已更新为: {value} 秒")
            self.analysis_interval = 1.0 / value
        elif key == "confidence_threshold":
            logger.info(f"置信度阈值已更新为: {value}")
            self.confidence_threshold = value
        elif key == "batch_size":
            logger.info(f"批处理大小已更新为: {value}")
            self.batch_size = value
        elif key == "enabled":
             logger.info(f"宏观表情引擎已{'启用' if value else '禁用'}")
             self.enabled = value
    
    def _on_system_config_changed(self, key: str, value: Any):
        """系统配置变更回调"""
        logger.debug(f"[_on_system_config_changed] Received system config change: {key} = {value}")
        self.system_config = self.config_manager.get('system', {})
        if key == "device":
            logger.info(f"设备已更新为: {value}")
            self.device = torch.device(value)
            if self.model:
                try:
                    self.model.to(self.device)
                    logger.info(f"模型已移至新设备: {self.device}")
                except Exception as e:
                    logger.error(f"移动模型到设备 {self.device} 失败: {e}")
        elif key == "models_dir":
            logger.info(f"模型目录已更新为: {value}")
            self.models_dir = value
            default_model_path = os.path.join(value, "macro.pt")
            if self.model_path == os.path.join(self.config_manager.get("system.models_dir", "models"), "macro.pt"):
                self.model_path = default_model_path
                self._load_model()
    
    def _on_face_detected(self, event: Event):
        """人脸检测事件处理"""
        logger.debug(f"[_on_face_detected] Received FACE_DETECTED event. Enabled: {self.config.get('enabled', True)}, Running: {self.is_running}, Paused: {self.is_paused}")
        if not self.config.get('enabled', True) or not self.is_running or self.is_paused:
            return
            
        # 提取事件数据
        event_data = event.data
        if not event_data or "face" not in event_data or "frame" not in event_data:
            return
            
        face = event_data["face"]
        frame = event_data["frame"]
        frame_id = event_data.get("frame_id", 0)
        
        # 在线程中进行情绪分析
        threading.Thread(
            target=self._analyze_emotion_threaded,
            args=(face, frame, frame_id),
            daemon=True
        ).start()
    
    def _analyze_emotion_threaded(self, face, frame, frame_id):
        """在线程中执行情绪分析"""
        logger.debug(f"[_analyze_emotion_threaded] Started for face_id: {face.face_id}, frame_id: {frame_id}")
        try:
            # 获取主模型的预测结果
            primary_result = self.analyze_face(frame, face) 
            final_result = primary_result # 默认使用主模型结果
            
            if final_result: 
                logger.debug(f"[_analyze_emotion_threaded] Primary analysis successful for face_id: {face.face_id}. Result: {final_result}")
                
                # --- 移除旧的 AU 辅助逻辑 ---
                # The AU assistance and suggestion logic has been moved to MacroEmotionAUEngine,
                # which subscribes to the event published below.
                # This engine (MacroEmotionEngine) is now responsible only for the primary detection.
                # --- 结束移除 ---
 
                # 发布原始的宏观情绪分析结果事件
                # 使用 RAW_MACRO_EMOTION_ANALYZED 以便 MacroEmotionAUEngine 可以拾取
                self.event_bus.publish(EventType.RAW_MACRO_EMOTION_ANALYZED, { # 修改事件类型
                    "face_id": face.face_id,
                    "frame_id": frame_id,
                    "result": final_result, # 这是主引擎的原始结果
                    "frame_data": {"image": frame, "timestamp": time.time()} # 可选：传递帧数据
                })
                
                 # 更新当前结果 (使用最终结果)
                with self.result_lock:
                    self.current_result = final_result # current_result 现在存储的是原始结果
            else:
                logger.debug(f"[_analyze_emotion_threaded] Primary analysis failed or returned None for face_id: {face.face_id}.")

        except Exception as e:
            logger.error(f"情绪分析线程错误 for face_id {face.face_id}: {e}", exc_info=True) # 添加 face_id 和 exc_info
    
    def _load_model(self):
        """加载表情识别模型"""
        logger.debug("[_load_model] Attempting to load model...")
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件未找到: {self.model_path}")
            self.model = None
            return
        
        try:
            logger.debug(f"[_load_model] Loading state_dict from: {self.model_path}")
            # 创建模型实例
            self.model = EmotionResNet(num_classes=len(EMOTION_CLASSES))
            
            # 加载模型
            logger.info(f"加载宏观表情模型: {self.model_path}")
            
            # 确定设备 (加载 checkpoint 时先统一加载到 CPU，避免潜在问题)
            map_location = torch.device('cpu') # Load to CPU first
            
            logger.debug(f"[_load_model] Loading checkpoint from {self.model_path} to CPU.")
            # 加载整个 checkpoint
            checkpoint = torch.load(self.model_path, map_location=map_location)
            
            # --- 提取 state_dict ---
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.debug("[_load_model] Extracted 'model_state_dict' from checkpoint.")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.debug("[_load_model] Extracted 'state_dict' from checkpoint.")
                else:
                    # 可能是直接保存的 state_dict 字典
                    state_dict = checkpoint
                    logger.debug("[_load_model] Checkpoint is a dict, assuming it's the state_dict itself.")
            elif isinstance(checkpoint, (OrderedDict, dict)): # 兼容直接保存的 state_dict
                state_dict = checkpoint
                logger.debug("[_load_model] Checkpoint is an OrderedDict/dict, assuming it's the state_dict itself.")
            else:
                # 可能是保存的整个模型实例
                logger.warning("[_load_model] Checkpoint might be a full model instance, not a state_dict. Trying to access state_dict().")
                if hasattr(checkpoint, 'state_dict') and callable(checkpoint.state_dict):
                    state_dict = checkpoint.state_dict()
                else:
                    logger.error("[_load_model] Failed to extract state_dict from the loaded checkpoint.")
                    raise ValueError("无法从 checkpoint 中提取 state_dict")

            if state_dict:
                # --- 修正键前缀处理 ---
                final_state_dict = OrderedDict()
                module_prefix_to_remove = "module."
                keys_processed = 0
                module_prefix_removed_count = 0
                
                for k, v in state_dict.items():
                    original_key = k
                    # 只移除 module. 前缀
                    if k.startswith(module_prefix_to_remove):
                        k = k[len(module_prefix_to_remove):]
                        final_state_dict[k] = v
                        logger.debug(f"[_load_model] Removed '{module_prefix_to_remove}' prefix: '{original_key}' -> '{k}'")
                        module_prefix_removed_count += 1
                    else:
                        final_state_dict[k] = v # 保留原始键名 (可能包含 resnet.)
                    keys_processed += 1
                    
                logger.info(f"[_load_model] Processed {keys_processed} keys in state_dict (removed '{module_prefix_to_remove}' prefix if present). {module_prefix_removed_count} keys modified.")
                # --- 结束修正键前缀处理 ---

                # 加载状态字典到模型 (模型此时在 CPU上)
                logger.debug("[_load_model] Attempting to load processed state_dict into model (strict=True)...")
                try:
                    # 优先尝试 strict=True，因为模型定义应该匹配
                    load_result = self.model.load_state_dict(final_state_dict, strict=True)
                    logger.debug(f"[_load_model] load_state_dict (strict=True) result: {load_result}")
                except RuntimeError as load_error:
                    logger.error(f"[_load_model] RuntimeError during load_state_dict (strict=True): {load_error}")
                    logger.warning("[_load_model] Trying load_state_dict with strict=False... (This might indicate a persistent model definition mismatch!)")
                    try:
                        load_result = self.model.load_state_dict(final_state_dict, strict=False)
                        logger.warning(f"[_load_model] Successfully loaded with strict=False. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
                        if load_result.missing_keys or load_result.unexpected_keys:
                            logger.error("模型定义与 Checkpoint 仍不完全匹配，即使 strict=False！引擎功能可能严重受限！请务必核对模型定义和 checkpoint 文件。")
                    except Exception as fallback_load_error:
                        logger.error(f"[_load_model] Failed to load state_dict even with strict=False: {fallback_load_error}", exc_info=True)
                        raise fallback_load_error # 再次尝试失败，抛出异常

                # 将加载了权重的模型，整体移动到目标设备
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"宏观表情模型已成功加载到设备 {self.device}: {self.model_path}")
            else:
                logger.error("[_load_model] state_dict is empty or None after extraction attempt.")
                raise ValueError("未能成功提取 state_dict")

        except Exception as e:
            logger.error(f"加载宏观表情模型权重或移动设备时失败: {e}", exc_info=True)
            self._initialized = False
            raise # 重新抛出异常，以便上层知道失败了
    
    # --- Helper function for warming up the model (optional) ---
    # def _warmup_model(self):
    #      if self.model is not None:
    #           logger.debug("[_warmup_model] Warming up model...")
    #           try:
    #                # Create a dummy input tensor matching the expected size
    #                dummy_input = torch.randn(1, 1, self.input_size[0], self.input_size[1]).to(self.device)
    #                with torch.no_grad():
    #                     _ = self.model(dummy_input)
    #                logger.debug("[_warmup_model] Model warmup complete.")
    #           except Exception as e:
    #                logger.warning(f"[_warmup_model] Model warmup failed: {e}")
    
    def preprocess_image(self, face_img):
        """预处理人脸图像用于模型输入"""
        try:
            # 确保图像存在
            if face_img is None:
                return None
                
            # 调整大小
            face_img = cv2.resize(face_img, self.input_size)
            
            # 确保图像是单通道灰度图
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            elif len(face_img.shape) == 2:
                pass # 已经是灰度图
            else:
                logger.warning(f"不支持的图像通道数: {face_img.shape}，将尝试转换为灰度图")
                # 尝试转换为灰度图，如果失败则返回 None
                try:
                     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY if len(face_img.shape) == 3 else cv2.COLOR_BGRA2GRAY)
                except cv2.error as e:
                    logger.error(f"转换为灰度图失败: {e}")
                    return None
            
            # 添加通道维度 (H, W) -> (H, W, C=1)
            face_img = face_img[..., np.newaxis]
            
            # 归一化 (灰度图只需单值)
            face_img = face_img.astype(np.float32) / 255.0
            
            # 标准化 (使用训练时的 mean=0.5, std=0.5)
            mean = 0.5
            std = 0.5
            face_img = (face_img - mean) / std
            
            # 转换为PyTorch张量 (H, W, C) -> (C, H, W) 并添加批维度
            face_tensor = torch.from_numpy(face_img.transpose(2, 0, 1)).unsqueeze(0)
            
            return face_tensor
            
        except Exception as e:
            logger.error(f"预处理人脸图像失败: {e}")
            return None
    
    def predict_emotion(self, face_img) -> Optional[EmotionResult]:
        """
        预测人脸图像的情绪
        
        Args:
            face_img: 人脸图像
            
        Returns:
            EmotionResult: 情绪分析结果
        """
        if self.model is None or face_img is None:
            return None
        
        try:
            # 计算图像哈希值，用于缓存检查
            img_hash = hash(face_img.tobytes())
            
            # 检查缓存
            if img_hash in self.results_cache:
                return self.results_cache[img_hash]
            
            # 计时开始
            start_time = time.time()
            
            # 预处理图像
            face_tensor = self.preprocess_image(face_img)
            if face_tensor is None:
                return None
            
            # 将张量移至设备 (确保与模型在同一设备)
            face_tensor = face_tensor.to(self.device)
            
            # 禁用梯度计算以加速推理
            with torch.no_grad():
                # --- 再次强制确保模型和输入都在同一设备 (就在调用前) ---
                self.model.to(self.device)       # 确保模型在目标设备
                face_tensor = face_tensor.to(self.device) # 再次确保输入在目标设备
                
                # --- 添加详细设备日志 ---
                model_device = next(self.model.parameters()).device # 获取模型参数实际所在的设备
                input_device = face_tensor.device
                logger.debug(f"[Predict Emotion] Target Device: {self.device}, Model Param Device: {model_device}, Input Tensor Device: {input_device}")
                if model_device != input_device or model_device != self.device:
                     logger.warning(f"[Predict Emotion] Device mismatch detected right before model call! Target: {self.device}, Model: {model_device}, Input: {input_device}")
                # --- 结束详细设备日志 ---
                
                # 前向传播
                output = self.model(face_tensor)
                
                # 应用softmax获取概率
                probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
            
            # 获取最高概率的情绪类别
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            
            # 创建情绪结果
            emotion_type = EMOTION_CLASSES.get(max_prob_idx, EmotionType.UNKNOWN)
            emotion_result = EmotionResult(
                emotion_type=emotion_type,
                probability=float(max_prob)
            )
            
            # 更新处理时间
            self.processing_time = time.time() - start_time
            
            # 记录当前结果
            with self.result_lock:
                self.current_result = emotion_result
            
            # 缓存结果
            self.results_cache[img_hash] = emotion_result
            
            # 限制缓存大小
            if len(self.results_cache) > self.max_cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.results_cache))
                del self.results_cache[oldest_key]
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"预测情绪失败: {e}")
            return None
    
    def analyze_face(self, face_img, face_detection: Optional[FaceDetection] = None) -> Optional[EmotionResult]:
        """
        分析人脸图像的情绪
        
        Args:
            face_img: 输入的人脸图像
            face_detection: 人脸检测结果
            
        Returns:
            EmotionResult: 情绪分析结果
        """
        if not self.is_running or self.is_paused or face_img is None:
            return None
            
        try:
            # 如果提供了人脸检测结果，使用检测到的人脸区域
            if face_detection and face_detection.face_box:
                # 如果有提取好的人脸图像，直接使用
                if face_detection.face_chip is not None:
                    face_crop = face_detection.face_chip
                else:
                    # 否则从原图裁剪人脸区域
                    x1, y1, x2, y2 = map(int, face_detection.face_box.to_tlbr())
                    face_crop = face_img[y1:y2, x1:x2]
            else:
                # 没有人脸检测结果，使用整个图像
                face_crop = face_img
            
            # 确保提取的人脸图像有效
            if face_crop is None or face_crop.size == 0:
                logger.warning("无效的人脸图像")
                return None
            
            # 预测情绪
            return self.predict_emotion(face_crop)
            
        except Exception as e:
            logger.error(f"分析人脸情绪失败: {e}")
            return None
    
    def start(self):
        """启动情绪分析引擎"""
        if self.is_running:
            logger.warning("宏观表情分析引擎已经在运行中")
            return
        
        self.is_running = True
        self.is_paused = False
        logger.info("宏观表情分析引擎已启动")
        
        # 通知引擎启动事件
        self.event_bus.publish(EventType.ENGINE_STARTED, {"engine": "macro"})
    
    def stop(self):
        """停止情绪分析引擎"""
        if not self.is_running:
            logger.warning("宏观表情分析引擎未运行")
            return
        
        self.is_running = False
        
        # 等待处理线程结束
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        logger.info("宏观表情分析引擎已停止")
        
        # 通知引擎停止事件
        self.event_bus.publish(EventType.ENGINE_STOPPED, {"engine": "macro"})
    
    def pause(self):
        """暂停情绪分析引擎"""
        if not self.is_running:
            logger.warning("宏观表情分析引擎未运行")
            return
        
        self.is_paused = True
        logger.info("宏观表情分析引擎已暂停")
        
        # 通知引擎暂停事件
        self.event_bus.publish(EventType.ENGINE_PAUSED, {"engine": "macro"})
    
    def resume(self):
        """恢复情绪分析引擎"""
        if not self.is_running:
            logger.warning("宏观表情分析引擎未运行")
            return
        
        self.is_paused = False
        logger.info("宏观表情分析引擎已恢复")
        
        # 通知引擎恢复事件
        self.event_bus.publish(EventType.ENGINE_RESUMED, {"engine": "macro"})
    
    def get_current_result(self) -> Optional[EmotionResult]:
        """获取当前情绪分析结果"""
        with self.result_lock:
            return self.current_result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计数据"""
        return {
            "processing_time": self.processing_time,
            "fps": 1.0 / self.processing_time if self.processing_time > 0 else 0.0,
            "cache_size": len(self.results_cache)
        }
    
    def clear_cache(self):
        """清除缓存的结果"""
        self.results_cache.clear()
        logger.info("结果缓存已清除")
    
    def draw_result(self, image: np.ndarray, face_detection: Optional[FaceDetection] = None) -> np.ndarray:
        """
        在图像上绘制情绪分析结果
        
        Args:
            image: 输入图像
            face_detection: 人脸检测结果
            
        Returns:
            np.ndarray: 带有可视化结果的图像
        """
        if image is None:
            return image
        
        # 获取当前情绪分析结果
        emotion_result = self.get_current_result()
        if emotion_result is None:
            return image
        
        # 创建可视化图像副本
        vis_img = image.copy()
        
        # 如果有人脸检测结果，在人脸上显示情绪
        if face_detection and face_detection.face_box:
            x1, y1, x2, y2 = map(int, face_detection.face_box.to_tlbr())
            
            # 绘制情绪标签
            emotion_text = f"{emotion_result.emotion_type.value}: {emotion_result.probability:.2f}"
            cv2.putText(vis_img, emotion_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_img