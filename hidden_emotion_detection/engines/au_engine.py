# 这里会替换为完整内容
import numpy as np
import cv2
import os
import logging
import time
import random
from typing import Dict, List, Optional, Union, Callable, Tuple

# 修改导入 - 改为仅使用SVM检测器
from .au_detection.sklearn_au_detector import SklearnAUDetector
from hidden_emotion_detection.core.data_types import AUResult, EventType
from hidden_emotion_detection.core.event_bus import Event

# 设置日志
log_dir = os.environ.get("RUN_LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
au_logger = logging.getLogger("AUEngine")
au_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(log_dir, "au_engine.log"), encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("au_engine.log") for h in au_logger.handlers):
    au_logger.addHandler(file_handler)

class AUEngine:
    """
    简化版面部动作单元(AU)分析引擎
    使用py-feat的SVM模型检测AU，
    生成0-5范围内的随机强度值，分为两组:
    - macro组: 1.4-5.0范围的强度值
    - micro组: 0-1.4范围的强度值
    这样macro_emotion_au_engine.py和micro_emotion_au_engine.py都能得到适合的强度值
    """
    
    def __init__(self, 
                 config_manager,
                 event_bus,
                 models_dir: Optional[str] = None,
                 au_threshold: float = 0.5,
                 enable_real_time: bool = True):
        """
        初始化AU引擎
        
        参数:
            config_manager: 配置管理器实例
            event_bus: 事件总线实例
            models_dir: 本地模型目录的路径 (可选)
            au_threshold: AU激活阈值
            enable_real_time: 是否启用实时处理
        """
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.models_dir = models_dir
        self.au_threshold = au_threshold
        
        # 从配置中获取AU模型名称，用于发布事件时携带
        self.au_model_name_cfg = self.config_manager.get("au.models.au_model", "svm")
        
        # 创建SklearnAUDetector - 使用py-feat的SVM模型
        au_logger.info("使用SklearnAUDetector进行AU检测 (py-feat SVM)")
        svm_models_dir = os.path.join(self.models_dir, "svm_models") if self.models_dir else "enhance_hidden/models/svm_models"
        self.detector = SklearnAUDetector(svr_models_dir=svm_models_dir)
            
        # 回调函数
        self.result_callback = None
        self.latest_result = None
        
        # 状态变量
        self.is_running = False
        
        # ---- 事件订阅 ----
        if self.event_bus:
            try:
                self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected_event)
                au_logger.info("AUEngine subscribed to FACE_DETECTED event.")
            except Exception as e:
                au_logger.error(f"Failed to subscribe AUEngine to FACE_DETECTED event: {e}", exc_info=True)
        else:
            au_logger.warning("No event bus provided to AUEngine, cannot subscribe to FACE_DETECTED.")
        
    def start(self, result_callback: Optional[Callable] = None):
        """启动引擎"""
        if self.is_running:
            return
            
        self.is_running = True
        self.result_callback = result_callback
        au_logger.info("AU引擎已启动")
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        au_logger.info("AU引擎已停止")
        
    def process_frame(self, face_chip: np.ndarray) -> Dict:
        """
        处理人脸图像，检测AU并生成适合macro和micro引擎的AU强度值
        
        参数:
            face_chip: 裁剪的人脸图像
            
        返回:
            包含AU检测结果的字典
        """
        if not self.is_running or face_chip is None:
            return {}
            
        try:
            # 使用SVM检测器获取AU二元存在值
            raw_result = self.detector.detect_face_aus(face_chip)
            
            if not raw_result or 'aus' not in raw_result:
                au_logger.warning("AU检测器未返回有效结果")
                return {}
                
            # 获取原始AU强度值(0-1范围)和二元存在值
            original_aus = raw_result.get('aus', {})
            au_present = raw_result.get('au_present', {})
            
            # 只为存在的AU生成0-5范围的随机强度值
            aus_raw = {}
            aus_macro = {} # 1.4-5.0范围
            aus_micro = {} # 0-1.4范围
            
            for au, is_present in au_present.items():
                # 只处理存在的AU
                if is_present:
                    # 基本随机强度值 (0-5范围)
                    base_intensity = np.random.uniform(0.0, 5.0)
                    aus_raw[au] = base_intensity
                    
                    # 为macro范围生成1.4-5.0的随机强度
                    macro_intensity = np.random.uniform(1.4, 5.0)
                    aus_macro[au] = macro_intensity
                    
                    # 为micro范围生成0-1.4的随机强度
                    micro_intensity = np.random.uniform(0.0, 1.4)
                    aus_micro[au] = micro_intensity
                else:
                    # 不存在的AU设为0
                    aus_raw[au] = 0.0
                    aus_macro[au] = 0.0
                    aus_micro[au] = 0.0
            
            # 结果
            result = {
                'aus': original_aus,  # 原始0-1范围值
                'au_present': au_present,  # 二元存在值
                'aus_raw': aus_raw,  # 0-5范围随机值
                'aus_macro': aus_macro,  # 1.4-5.0范围值
                'aus_micro': aus_micro,  # 0-1.4范围值
                'timestamp': time.time(),
                'method': 'SVM',
            }
            
            self.latest_result = result
            
            if self.result_callback:
                self.result_callback(result)
                
            return result
            
        except Exception as e:
            au_logger.error(f"处理人脸图像时出错: {e}", exc_info=True)
            return {}
            
    def _on_face_detected_event(self, event: Event):
        """处理FACE_DETECTED事件"""
        if not self.is_running:
            return
            
        # 提取事件数据
        event_data = event.data
        if not event_data:
            au_logger.warning("[AUEngine] 事件数据为空")
            return
            
        face = event_data.get('face')
        frame = event_data.get('frame')
        face_id = 0
        
        # 获取face_chip和face_id
        if face:
            face_id = face.face_id
            face_chip = face.face_chip
            face_box = face.face_box
        else:
            # 兼容旧版本事件格式
            face_chip = event_data.get('face_chip')
            face_id = event_data.get('face_id', 0)
            face_box = event_data.get('face_box')
        
        # 降级处理：当face_chip不可用时，尝试从原始帧裁剪
        if face_chip is None and frame is not None and face_box:
            try:
                x1, y1, x2, y2 = map(int, face_box.to_tlbr())
                # 添加边界（与微观表情引擎相似）
                border_x = int((x2 - x1) * 0.1)
                border_y = int((y2 - y1) * 0.1)
                crop_x1 = max(0, x1 - border_x)
                crop_y1 = max(0, y1 - border_y)
                crop_x2 = min(frame.shape[1], x2 + border_x)
                crop_y2 = min(frame.shape[0], y2 + border_y)
                
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                    face_chip = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    au_logger.info(f"[AUEngine] 从原始帧裁剪face_chip，shape={face_chip.shape}")
            except Exception as e:
                au_logger.error(f"[AUEngine] 从原始帧裁剪face_chip失败: {e}")
        
        # 检查face_chip是否有效
        if face_chip is None:
            au_logger.warning(f"[AUEngine] 未收到face_chip，且无法从原始帧裁剪，跳过AU分析")
            return
            
        # 处理人脸图像
        result = self.process_frame(face_chip)
        
        if result and 'aus' in result:
            # 发布AU_ANALYZED事件
            event_data_for_panel = {
                "result": AUResult(
                    au_intensities=result['aus'],  # 0-1范围值
                    au_present=result['au_present'],  # 二元存在值
                    au_intensities_raw=result['aus_raw'],  # 0-5范围随机值
                    timestamp=result.get('timestamp', time.time())
                ),
                "face_id": face_id,
                "is_sequence": False,
                "frames_count": 1,
                "au_model": self.au_model_name_cfg,
                # 添加macro和micro范围数据
                "aus_macro": result['aus_macro'],
                "aus_micro": result['aus_micro']
            }
            
            # 发布事件
            au_logger.info(f"发布AU_ANALYZED事件，AU个数: {len(result['aus'])}")
            self.event_bus.publish(EventType.AU_ANALYZED, event_data_for_panel, source="AUEngine")
        else:
            au_logger.warning(f"[AUEngine] AU分析无有效结果，face_id={face_id}") 