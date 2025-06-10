"""
分析引擎层模块
包含表情分析核心引擎：宏观表情、微表情、AU分析和隐藏情绪分析
"""

# 导入宏观表情引擎
try:
    from .macro_emotion_engine import MacroEmotionEngine
except ImportError:
    import logging
    logging.warning("宏观表情引擎导入失败")
    MacroEmotionEngine = None

# 导入微表情引擎
try:
    from .micro_emotion_engine import MicroEmotionEngine
except ImportError:
    import logging
    logging.warning("微表情引擎导入失败")
    MicroEmotionEngine = None

# 导入AU分析引擎
try:
    from .au_engine import AUEngine
except ImportError as e:
    import logging
    logging.warning("AU分析引擎导入失败")
    logging.error(f"导入失败: {e}")
    AUEngine = None

# 导入隐藏情绪引擎
try:
    from .hidden_emotion_engine import HiddenEmotionEngine
except ImportError:
    import logging
    logging.warning("隐藏情绪引擎导入失败")
    HiddenEmotionEngine = None

# 导入人脸检测引擎
try:
    from .face_detection_engine import FaceDetectionEngine
except ImportError:
    import logging
    logging.warning("人脸检测引擎导入失败")
    FaceDetectionEngine = None

# 导入姿态估计引擎
try:
    from .pose_estimator import PoseEstimator
except ImportError:
    import logging
    logging.warning("姿态估计引擎导入失败")
    PoseEstimator = None

# 根据配置选择引擎实现
SpeechToTextEngine = None

__all__ = [
    'MacroEmotionEngine',
    'MicroEmotionEngine',
    'AUEngine',
    'HiddenEmotionEngine',
    'FaceDetectionEngine',
    'PoseEstimator',
    'SpeechToTextEngine'
] 