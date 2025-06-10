#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据类型定义模块
定义系统中使用的所有标准数据结构，确保模块间通信的一致性
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import time
from enum import Enum, auto
import cv2

# 情绪类型枚举
class EmotionType(Enum):
    """面部表情枚举类型"""
    NEUTRAL = 0     # 中性表情
    HAPPINESS = 1   # 高兴
    SADNESS = 2     # 悲伤
    SURPRISE = 3    # 惊讶
    FEAR = 4        # 恐惧
    ANGER = 5       # 愤怒
    DISGUST = 6     # 厌恶
    CONTEMPT = 7    # 蔑视
    CONFUSION = 8   # 困惑/压抑
    REPRESSION = 10 # 新增：压抑情绪
    UNKNOWN = 9     # 未知表情
    
    def __str__(self):
        """获取情绪类型的字符串表示"""
        emotion_names = {
            EmotionType.NEUTRAL: "中性",
            EmotionType.HAPPINESS: "高兴",
            EmotionType.SADNESS: "悲伤",
            EmotionType.SURPRISE: "惊讶",
            EmotionType.FEAR: "恐惧",
            EmotionType.ANGER: "愤怒",
            EmotionType.DISGUST: "厌恶",
            EmotionType.CONTEMPT: "蔑视",
            EmotionType.CONFUSION: "压抑",
            EmotionType.REPRESSION: "压抑",
            EmotionType.UNKNOWN: "未知"
        }
        return emotion_names.get(self, "未知")

class FaceDetection:
    """面部检测结果类"""
    
    def __init__(self, 
                 x: int = 0, 
                 y: int = 0, 
                 width: int = 0, 
                 height: int = 0,
                 confidence: float = 0.0,
                 landmarks: List[Tuple[int, int]] = None,
                 face_id: int = -1,
                 timestamp: float = None):
        """
        初始化面部检测结果
        
        Args:
            x: 人脸框左上角x坐标
            y: 人脸框左上角y坐标
            width: 人脸框宽度
            height: 人脸框高度
            confidence: 检测置信度
            landmarks: 面部特征点列表，格式为[(x1,y1), (x2,y2), ...]
            face_id: 人脸ID，用于跟踪
            timestamp: 时间戳
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.landmarks = landmarks if landmarks is not None else []
        self.face_id = face_id
        self.timestamp = timestamp if timestamp is not None else time.time()
        
        # 姿态信息
        self.pitch = 0.0  # 俯仰角
        self.yaw = 0.0    # 偏航角
        self.roll = 0.0   # 翻滚角
        
        # 提取的脸部图像
        self.face_image = None
        
    def get_center(self) -> Tuple[int, int]:
        """获取人脸中心点坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def get_rect(self) -> Tuple[int, int, int, int]:
        """获取人脸矩形框"""
        return (self.x, self.y, self.width, self.height)
    
    def get_area(self) -> int:
        """获取人脸面积"""
        return self.width * self.height
    
    def set_pose(self, pitch: float, yaw: float, roll: float):
        """设置姿态信息"""
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
    
    def set_face_image(self, image):
        """设置提取的脸部图像"""
        self.face_image = image

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "face_id": self.face_id,
            "landmarks": self.landmarks,
            "timestamp": self.timestamp,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceDetection':
        """从字典创建实例"""
        face = cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            confidence=data.get("confidence", 0.0),
            landmarks=data.get("landmarks", []),
            face_id=data.get("face_id", -1),
            timestamp=data.get("timestamp", time.time())
        )
        
        # 设置姿态
        face.set_pose(
            data.get("pitch", 0.0),
            data.get("yaw", 0.0),
            data.get("roll", 0.0)
        )
        
        return face

@dataclass
class FaceBox:
    """人脸边界框"""
    x1: float  # 左上角x坐标
    y1: float  # 左上角y坐标
    x2: float  # 右下角x坐标
    y2: float  # 右下角y坐标
    confidence: float = 0.0  # 置信度
    
    @property
    def width(self) -> float:
        """获取宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """获取高度"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """获取面积"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """获取中心点坐标"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """转换为top-left-width-height格式"""
        return (self.x1, self.y1, self.width, self.height)
    
    def to_tlbr(self) -> Tuple[float, float, float, float]:
        """转换为top-left-bottom-right格式"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """转换为center-x-y-width-height格式"""
        return (self.center[0], self.center[1], self.width, self.height)
    
    def to_opencv_rect(self) -> Tuple[int, int, int, int]:
        """转换为OpenCV矩形格式 (x, y, w, h)"""
        return (int(self.x1), int(self.y1), int(self.width), int(self.height))
    
    def iou(self, other: 'FaceBox') -> float:
        """计算与另一个边界框的IoU"""
        # 计算交集区域
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        # 如果不存在交集，则返回0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集面积
        union_area = self.area + other.area - intersection_area
        
        # 返回IoU
        return intersection_area / union_area if union_area > 0 else 0.0

@dataclass
class FacePose:
    """面部姿态"""
    pitch: float = 0.0  # 俯仰角（上下点头）
    yaw: float = 0.0    # 偏航角（左右摇头）
    roll: float = 0.0   # 翻滚角（头部倾斜）
    # Add fields for translation vector
    x_pos: Optional[float] = None # Estimated X position (e.g., mm)
    y_pos: Optional[float] = None # Estimated Y position
    z_pos: Optional[float] = None # Estimated Z position (depth)
    # Add rvec and tvec for 3D box drawing
    rvec: Optional[np.ndarray] = None # Rotation vector from solvePnP
    tvec: Optional[np.ndarray] = None # Translation vector from solvePnP
    
    def is_frontal(self, threshold: float = 30.0) -> bool:
        """判断是否为正面姿态（角度小于阈值）"""
        return (abs(self.pitch) < threshold and 
                abs(self.yaw) < threshold and 
                abs(self.roll) < threshold)
    
    def get_dominant_angle(self) -> Tuple[str, float]:
        """获取主要偏转角度及其类型"""
        angles = {
            "pitch": abs(self.pitch),
            "yaw": abs(self.yaw),
            "roll": abs(self.roll)
        }
        max_type = max(angles, key=angles.get)
        return max_type, angles[max_type]
    
    def __str__(self) -> str:
        """获取姿态的字符串表示"""
        return f"俯仰={self.pitch:.1f}°, 偏航={self.yaw:.1f}°, 翻滚={self.roll:.1f}°"

@dataclass
class Landmarks:
    """面部关键点"""
    points: np.ndarray  # 关键点坐标数组，形状为 (N, 2)
    confidence: np.ndarray = None  # 关键点置信度，形状为 (N,)
    
    def __post_init__(self):
        """后初始化：确保points是numpy数组"""
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points)
        
        if self.confidence is not None and not isinstance(self.confidence, np.ndarray):
            self.confidence = np.array(self.confidence)
    
    @property
    def num_points(self) -> int:
        """获取关键点数量"""
        return len(self.points)
    
    def get_point(self, index: int) -> Tuple[float, float]:
        """获取指定索引的关键点坐标"""
        if 0 <= index < self.num_points:
            return tuple(self.points[index])
        return None
    
    def get_points_by_indices(self, indices: List[int]) -> np.ndarray:
        """获取指定索引列表的关键点坐标"""
        return self.points[indices]
    
    def get_eye_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取眼睛区域的关键点（仅适用于68点模型）"""
        # 假设使用的是68点模型，左眼点36-41，右眼点42-47
        if self.num_points >= 68:
            left_eye = self.points[36:42]
            right_eye = self.points[42:48]
            return left_eye, right_eye
        return None, None
    
    def get_mouth_points(self) -> np.ndarray:
        """获取嘴部区域的关键点（仅适用于68点模型）"""
        # 假设使用的是68点模型，嘴部点48-67
        if self.num_points >= 68:
            return self.points[48:68]
        return None
    
    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
             radius: int = 1, thickness: int = -1) -> np.ndarray:
        """在图像上绘制关键点"""
        vis_img = image.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(vis_img, (int(x), int(y)), radius, color, thickness)
        return vis_img

@dataclass
class AUStateMap:
    """AU状态映射"""
    states: Dict[str, bool] = field(default_factory=dict)  # AU激活状态 {AU01: True, ...}
    intensities: Dict[str, float] = field(default_factory=dict)  # AU强度 {AU01: 0.75, ...}
    durations: Dict[str, float] = field(default_factory=dict)  # AU持续时间 {AU01: 1.5, ...}
    
    def is_active(self, au_name: str) -> bool:
        """检查AU是否激活"""
        return self.states.get(au_name, False)
    
    def get_intensity(self, au_name: str) -> float:
        """获取AU强度"""
        return self.intensities.get(au_name, 0.0)
    
    def get_duration(self, au_name: str) -> float:
        """获取AU持续时间"""
        return self.durations.get(au_name, 0.0)
    
    def get_active_aus(self) -> List[str]:
        """获取所有激活的AU"""
        return [au for au, active in self.states.items() if active]
    
    def get_dominant_aus(self, top_n: int = 3) -> Dict[str, float]:
        """获取强度最高的AU"""
        sorted_aus = sorted(
            self.intensities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return {k: v for k, v in sorted_aus[:top_n]}

@dataclass
class AUResult:
    """AU分析结果"""
    au_intensities: Dict[str, float] # Normalized intensities (0-1)
    au_present: Dict[str, bool]
    au_intensities_raw: Optional[Dict[str, float]] = None # Raw intensities (0-5) from OpenFace
    timestamp: Optional[float] = None

@dataclass
class FaceDetection:
    """人脸检测结果"""
    face_box: FaceBox  # 人脸边界框
    landmarks: Optional[Landmarks] = None  # 面部关键点
    pose: Optional[FacePose] = None  # 面部姿态
    face_chip: Optional[np.ndarray] = None  # 裁剪的面部图像
    confidence: float = 0.0  # 检测置信度
    face_id: int = -1  # 人脸ID（用于追踪）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def draw(self, image: np.ndarray, 
             draw_box: bool = True, 
             draw_landmarks: bool = True,
             draw_pose: bool = True,
             box_color: Tuple[int, int, int] = (0, 255, 0),
             landmark_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """在图像上绘制检测结果"""
        vis_img = image.copy()
        
        # 绘制人脸边界框
        if draw_box and self.face_box:
            x1, y1, x2, y2 = self.face_box.to_tlbr()
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # 绘制人脸ID和置信度
            label = f"ID:{self.face_id}, Conf:{self.confidence:.2f}"
            cv2.putText(vis_img, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # 绘制面部关键点
        if draw_landmarks and self.landmarks:
            vis_img = self.landmarks.draw(vis_img, color=landmark_color)
        
        # 绘制姿态信息
        if draw_pose and self.pose:
            x1, y1 = self.face_box.to_tlbr()[:2]
            pose_text = str(self.pose)
            cv2.putText(vis_img, pose_text, (int(x1), int(y1) - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        return vis_img
    
    def get_face_chip(self, image: np.ndarray, size: Tuple[int, int] = (96, 96)) -> np.ndarray:
        """从原始图像中提取面部区域"""
        if self.face_chip is not None:
            # 如果已经有裁剪的面部，直接返回
            if self.face_chip.shape[:2] == size:
                return self.face_chip
            else:
                return cv2.resize(self.face_chip, size)
        
        # 否则从原图中裁剪
        x1, y1, x2, y2 = map(int, self.face_box.to_tlbr())
        # 确保边界在图像范围内
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 裁剪并缩放
        if x1 < x2 and y1 < y2:
            face_region = image[y1:y2, x1:x2]
            return cv2.resize(face_region, size)
        
        # 如果裁剪失败，返回空图像
        return np.zeros((*size, 3), dtype=np.uint8)

@dataclass
class EmotionResult:
    """情绪分析结果"""
    emotion_type: EmotionType  # 情绪类型
    probability: float  # 概率/置信度
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def __str__(self) -> str:
        """获取情绪结果的字符串表示"""
        return f"{self.emotion_type}({self.probability:.2f})"

@dataclass
class HiddenEmotionResult:
    """隐藏情绪分析结果"""
    surface_emotion: EmotionType  # 表面情绪
    hidden_emotion: EmotionType  # 隐藏情绪
    surface_prob: float  # 表面情绪概率
    hidden_prob: float  # 隐藏情绪概率
    conflict_score: float  # 冲突程度
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def has_conflict(self, threshold: float = 0.3) -> bool:
        """判断是否存在情绪冲突"""
        return self.conflict_score >= threshold
    
    def __str__(self) -> str:
        """获取隐藏情绪结果的字符串表示"""
        return f"表面:{self.surface_emotion}({self.surface_prob:.2f}), " \
               f"隐藏:{self.hidden_emotion}({self.hidden_prob:.2f}), " \
               f"冲突:{self.conflict_score:.2f}"

@dataclass
class FaceResult:
    """面部分析综合结果"""
    face_id: int  # 人脸ID
    face_detection: FaceDetection  # 人脸检测结果
    emotion_result: Optional[EmotionResult] = None  # 情绪分析结果
    au_result: Optional[AUResult] = None  # AU分析结果
    hidden_emotion_result: Optional[HiddenEmotionResult] = None  # 隐藏情绪分析结果
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def has_emotion(self) -> bool:
        """检查是否有情绪分析结果"""
        return self.emotion_result is not None
    
    def has_au(self) -> bool:
        """检查是否有AU分析结果"""
        return self.au_result is not None
    
    def has_hidden_emotion(self) -> bool:
        """检查是否有隐藏情绪分析结果"""
        return self.hidden_emotion_result is not None
    
    def get_emotion_type(self) -> Optional[EmotionType]:
        """获取情绪类型"""
        if self.emotion_result:
            return self.emotion_result.emotion_type
        return None
    
    def get_emotion_probability(self) -> float:
        """获取情绪概率"""
        if self.emotion_result:
            return self.emotion_result.probability
        return 0.0
    
    def get_hidden_emotion_type(self) -> Optional[EmotionType]:
        """获取隐藏情绪类型"""
        if self.hidden_emotion_result:
            return self.hidden_emotion_result.hidden_emotion
        return None
    
    def get_hidden_emotion_probability(self) -> float:
        """获取隐藏情绪概率"""
        if self.hidden_emotion_result:
            return self.hidden_emotion_result.hidden_prob
        return 0.0
    
    def get_conflict_score(self) -> float:
        """获取情绪冲突分数"""
        if self.hidden_emotion_result:
            return self.hidden_emotion_result.conflict_score
        return 0.0

@dataclass
class FrameResult:
    """帧处理结果"""
    frame_id: int  # 帧ID
    timestamp: float = field(default_factory=time.time)  # 时间戳
    face_detections: List[FaceDetection] = field(default_factory=list)  # 人脸检测结果
    au_results: Dict[int, AUResult] = field(default_factory=dict)  # 面部行为单元结果 {face_id: result}
    emotion_results: Dict[int, EmotionResult] = field(default_factory=dict)  # 情绪分析结果 {face_id: result}
    hidden_emotion_results: Dict[int, HiddenEmotionResult] = field(default_factory=dict)  # 隐藏情绪结果 {face_id: result}
    
    def has_faces(self) -> bool:
        """检查是否检测到人脸"""
        return len(self.face_detections) > 0
    
    def get_face_count(self) -> int:
        """获取人脸数量"""
        return len(self.face_detections)
    
    def get_face_by_id(self, face_id: int) -> Optional[FaceDetection]:
        """根据ID获取人脸检测结果"""
        for face in self.face_detections:
            if face.face_id == face_id:
                return face
        return None
    
    def get_au_result(self, face_id: int) -> Optional[AUResult]:
        """获取特定人脸的AU分析结果"""
        return self.au_results.get(face_id)
    
    def get_emotion_result(self, face_id: int) -> Optional[EmotionResult]:
        """获取特定人脸的情绪分析结果"""
        return self.emotion_results.get(face_id)
    
    def get_hidden_emotion_result(self, face_id: int) -> Optional[HiddenEmotionResult]:
        """获取特定人脸的隐藏情绪分析结果"""
        return self.hidden_emotion_results.get(face_id)
    
    def get_dominant_face(self) -> Optional[FaceDetection]:
        """获取主要人脸（通常是最大的或中心的）"""
        if not self.face_detections:
            return None
        
        # 按面积排序，返回最大的
        return sorted(self.face_detections, key=lambda x: x.face_box.area, reverse=True)[0]

# 事件类型定义
class EventType(Enum):
    """事件类型枚举"""
    FACE_DETECTED = auto()        # 人脸检测
    FACE_LOST = auto()            # 人脸丢失
    AU_ANALYZED = auto()          # AU分析完成
    AU_STATE_CHANGED = auto()     # AU状态变化
    EMOTION_DETECTED = auto()     # 情绪检测
    EMOTION_CHANGED = auto()      # 情绪变化
    
    RAW_MACRO_EMOTION_ANALYZED = auto() # 主宏观引擎原始结果
    AU_MACRO_EMOTION_ANALYZED = auto()  # AU辅助的宏观情绪分析结果
    MACRO_EMOTION_ANALYZED = auto()  # 宏观情绪分析 (可能被AU修正)
    
    RAW_MICRO_EMOTION_ANALYZED = auto() # 主微表情引擎原始结果
    AU_MICRO_EMOTION_ANALYZED = auto()  # AU辅助的微表情分析结果
    MICRO_EMOTION_ANALYZED = auto()  # 微观情绪分析 (可能被AU修正)
    
    HIDDEN_EMOTION_DETECTED = auto() # 隐藏情绪检测
    HIDDEN_EMOTION_ANALYZED = auto() # 隐藏情绪分析
    HIDDEN_EMOTION_CHANGED = auto()  # 隐藏情绪变化
    FRAME_PROCESSED = auto()      # 帧处理完成
    ENGINE_STARTED = auto()       # 引擎启动
    ENGINE_STOPPED = auto()       # 引擎停止
    ENGINE_PAUSED = auto()        # 引擎暂停
    ENGINE_RESUMED = auto()       # 引擎恢复
    CONFIG_CHANGED = auto()       # 配置变更
    ERROR_OCCURRED = auto()       # 错误发生
    USER_INPUT = auto()           # 用户输入
    SYSTEM_STATUS = auto()        # 系统状态

@dataclass
class Event:
    """事件数据类"""
    type: EventType  # 事件类型
    data: Any  # 事件数据
    source: str  # 事件源
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # 时间戳