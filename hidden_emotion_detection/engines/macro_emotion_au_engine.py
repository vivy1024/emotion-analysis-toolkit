# 这里会替换为完整内容
import numpy as np
import cv2
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import deque

from .au_engine import AUEngine
from .au_detection.analyzer import AUAnalyzer
from .au_detection.au_emotion_mapper import AUEmotionMapper
from hidden_emotion_detection.core.config_manager import ConfigManager
from hidden_emotion_detection.core.event_bus import EventBus, Event, EventType
from hidden_emotion_detection.core.data_types import EmotionResult, FrameData, AUResult, Emotion

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MacroEmotionAUEngine")

# 严格遵循备份文件 au_macro_emotion.py.backup 中的宏观表情AU映射规则
# AU强度值假设与备份文件中的尺度一致 (例如0-5范围)
# Lambda函数中的AU键已适配为数值字符串 (例如 '6', '12')
MACRO_EMOTION_AU_MAPPING = {
    Emotion.HAPPY: {
        'required': ['AU06', 'AU12'],
        'optional': [],
        'min_intensity': 1.4,
        'max_intensity': 5.0, # 用于潜在的范围检查，当前主要使用min_intensity
        'confidence_factor': 1.1,
        'coherence_check': {
            'rule': lambda au_vals: au_vals.get('6', 0) >= au_vals.get('12', 0) * 0.6,
            'penalty': 0.8
        }
    },
    Emotion.SADNESS: {
        'required': ['AU01', 'AU04'],
        'optional': ['AU15', 'AU17'],
        'min_intensity': 1.5,
        'max_intensity': 5.0,
        'confidence_factor': 1.0,
        'coherence_check': {
            # 'rule': lambda intensities: abs(intensities.get('AU01', 0) - intensities.get('AU15', 0) if 'AU15' in intensities else 0) < 1.0,
            # 适配后的lambda:
            'rule': lambda au_vals: abs(au_vals.get('1', 0) - (au_vals.get('15', 0) if '15' in au_vals else 0)) < 1.0,
            'penalty': 0.9
        }
    },
    Emotion.SURPRISE: {
        'required': ['AU01', 'AU02'],
        'optional': ['AU05', 'AU26'],
        'min_intensity': 1.4,
        'max_intensity': 5.0,
        'confidence_factor': 1.1
    },
    Emotion.FEAR: {
        'required': ['AU01', 'AU02', 'AU04', 'AU05'],
        'optional': ['AU20', 'AU26'],
        'min_intensity': 1.7,
        'max_intensity': 5.0,
        'confidence_factor': 0.9,
        'coherence_check': {
            # 'rule': lambda intensities: intensities.get('AU04', 0) >= 1.5,
            # 适配后的lambda:
            'rule': lambda au_vals: au_vals.get('4', 0) >= 1.5,
            'penalty': 0.75
        }
    },
    Emotion.DISGUST: {
        'required': ['AU09'],
        'optional': ['AU10', 'AU17'],
        'min_intensity': 1.8,
        'max_intensity': 5.0,
        'confidence_factor': 0.95
    },
    Emotion.ANGER: {
        'required': ['AU04'],
        'optional': ['AU05', 'AU07', 'AU23', 'AU24'],
        'min_intensity': 1.5,
        'max_intensity': 5.0,
        'confidence_factor': 1.0,
        'coherence_check': {
            # 'rule': lambda intensities: intensities.get('AU04', 0) >= 1.3,
            # 适配后的lambda:
            'rule': lambda au_vals: au_vals.get('4', 0) >= 1.3,
            'penalty': 0.9
        },
        'conflicts': {
            'AU12': 0.8 # 冲突AU的键也需要适配
        }
    },
    Emotion.CONTEMPT: {
        'required': ['AU12', 'AU14'], # Backup uses AU12, AU14. Contempt often unilateral.
        'optional': [],
        'min_intensity': 1.6,
        'max_intensity': 5.0,
        'confidence_factor': 0.9
    }
}

class MacroEmotionAUEngine:
    """
    宏观表情AU辅助分析引擎。
    接收主宏观表情引擎的分析结果，并结合AU数据进行验证、修正或增强。
    """
    
    def __init__(self, 
                 config_manager: ConfigManager,
                 event_bus: EventBus,
                 au_engine: Optional[AUEngine] = None):
        """
        初始化宏观表情AU辅助引擎
        
        参数:
            config_manager: 配置管理器实例
            event_bus: 事件总线实例
            au_engine: AU引擎实例。如果为None，将尝试从config或共享实例获取（暂定创建新的）。
        """
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.is_running = False
        self.frame_count = 0
    
        # 获取或创建AU引擎实例
        if au_engine:
            self.au_engine = au_engine
            logger.info("[MacroEmotionAUEngine] Using provided AUEngine instance.")
        else:
            logger.info("[MacroEmotionAUEngine] No AUEngine provided, creating a new one.")
            # 从配置中获取 models_dir (如果存在)
            models_dir_cfg = self.config_manager.get("au.models_dir", None) 
            
            self.au_engine = AUEngine(
                config_manager=self.config_manager,
                event_bus=self.event_bus,
                models_dir=models_dir_cfg,
                au_threshold=self.config_manager.get("au.threshold", 0.5),
                enable_real_time=self.config_manager.get("au.engine.enable_real_time", True)
            )
            logger.info("[MacroEmotionAUEngine] New AUEngine instance created.")

        # 从配置获取宏观情绪AU分析的最低置信度阈值
        self.minimum_confidence_threshold = self.config_manager.get(
            "macro_emotion_au.minimum_confidence_threshold", 0.07
        )
        
        # 添加事件订阅
        if self.event_bus:
            try:
                self.event_bus.subscribe(
                    EventType.RAW_MACRO_EMOTION_ANALYZED, 
                    self.handle_primary_macro_emotion_event
                )
                logger.info("[MacroEmotionAUEngine] Subscribed to RAW_MACRO_EMOTION_ANALYZED event.")
                
                # 额外订阅AU_ANALYZED事件，以便及时获取最新的AU数据
                self.event_bus.subscribe(
                    EventType.AU_ANALYZED,
                    self._on_au_analyzed
                )
                logger.info("[MacroEmotionAUEngine] Also subscribed to AU_ANALYZED event for latest data.")
            except Exception as e:
                logger.error(f"[MacroEmotionAUEngine] Failed to subscribe to events: {e}")
        else:
            logger.warning("[MacroEmotionAUEngine] Event bus not available, cannot subscribe to events.")
            
        # 存储最新的AU数据
        self.latest_au_data = {}
        self.latest_aus_macro = {}  # 专门存储宏观情绪AU数据(1.4-5.0范围)

        logger.info("MacroEmotionAUEngine initialized.")

    def start(self):
        """
        启动引擎。
        注意：此引擎主要由外部事件驱动 (例如，主宏观表情引擎的结果)，
        但其内部的 AUEngine 需要启动（如果是由它创建和管理的）。
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.frame_count = 0
        
        # 如果 AUEngine 是由这个类创建和管理的，确保它已启动
        if self.au_engine:
            # MacroEmotionAUEngine 不直接从 AUEngine 接收回调，它按需查询 AUEngine
            self.au_engine.start(result_callback=None) 
            logger.info("[MacroEmotionAUEngine] Internal AUEngine started.")
        
        logger.info("MacroEmotionAUEngine started and ready to process emotion results.")
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        if self.au_engine:
            self.au_engine.stop()
            logger.info("[MacroEmotionAUEngine] Internal AUEngine stopped.")
        logger.info("MacroEmotionAUEngine stopped.")
        
    def _on_au_analyzed(self, event: Event):
        """处理AU_ANALYZED事件，更新最新的AU数据"""
        try:
            if not self.is_running:
                return
                
            event_data = event.data
            if not isinstance(event_data, dict):
                logger.warning(f"[MacroEmotionAUEngine] Received AU_ANALYZED with unexpected data type: {type(event_data)}")
                return
                
            # 获取AUResult对象
            au_result = event_data.get('result')
            if au_result and hasattr(au_result, 'au_intensities'):
                # 保存标准化AU强度值(0-1范围)
                self.latest_au_data = au_result.au_intensities if au_result.au_intensities else {}
                
                # 获取原始非规范化强度值(如果存在)
                if hasattr(au_result, 'au_intensities_raw'):
                    raw_intensities = au_result.au_intensities_raw
                    if raw_intensities:
                        logger.debug(f"[MacroEmotionAUEngine] Received raw AU intensities: {raw_intensities}")
                
                # 保存宏观范围的AU值(1.4-5.0范围)，用于宏观情绪分析
                if 'aus_macro' in event_data:
                    self.latest_aus_macro = event_data['aus_macro']
                    logger.info(f"[MacroEmotionAUEngine] Updated macro AU data from event: {len(self.latest_aus_macro)} AUs, keys: {list(self.latest_aus_macro.keys())[:5]}...")
                    
                logger.debug(f"[MacroEmotionAUEngine] Updated AU data from event: {len(self.latest_au_data)} AUs")
        except Exception as e:
            logger.error(f"[MacroEmotionAUEngine] Error processing AU_ANALYZED event: {e}")
            logger.error(traceback.format_exc())

    def normalize_au_key(self, key):
        if isinstance(key, str) and key.startswith('AU'):
            # 去掉前缀并去除前导零，例如"AU06"变为"6"
            return key[2:].lstrip('0')
        # 如果已经是数字格式，则直接返回
        return key

    def suggest_alternative_or_confirm(
        self, 
        primary_result: EmotionResult, 
        frame_data: Optional[FrameData] = None, 
        current_aus_direct: Optional[Dict[str, float]] = None
    ) -> EmotionResult:
        """
        根据AU数据，对主宏观表情引擎的结果提供建议或确认。
        严格遵循备份文件中的AU映射和评分逻辑。
        """
        try:
            if not self.is_running:
                logger.warning("MacroEmotionAUEngine is not running. Returning primary result.")
                return primary_result
                
            logger.info(f"[MacroSuggestStrict] Processing primary emotion: {primary_result.emotion_type.name}, confidence: {primary_result.probability:.2f}")
    
            au_values: Optional[Dict[str, float]] = None
            
            # 1. 优先使用直接提供的AU数据
            if current_aus_direct and len(current_aus_direct) > 0:
                au_values = current_aus_direct
                logger.info(f"[MacroSuggestStrict] Using directly provided AUs: {list(au_values.keys()) if au_values else 'None'}")
            # 2. 尝试从frame_data获取AU数据
            elif frame_data and frame_data.image is not None:
                if self.au_engine:
                    au_engine_output = self.au_engine.process_frame(frame_data.image)
                    if au_engine_output and 'aus_macro' in au_engine_output:
                        # 优先使用宏观范围的AU强度值
                        au_values = au_engine_output['aus_macro']
                        logger.info(f"[MacroSuggestStrict] Using macro AU values from process_frame: {len(au_values)} AUs")
                    elif au_engine_output and 'aus' in au_engine_output:
                        # 退回到原始AU值
                        au_values = au_engine_output['aus']
                        logger.info(f"[MacroSuggestStrict] Using standard AU values from process_frame: {len(au_values)} AUs")
            # 3. 使用最新接收到的AU数据
            elif self.latest_aus_macro and len(self.latest_aus_macro) > 0:
                au_values = self.latest_aus_macro
                logger.info(f"[MacroSuggestStrict] Using latest macro AU data from event: {len(self.latest_aus_macro)} AUs")
            # 4. 回退到从AU引擎获取最新结果
            elif self.au_engine and self.au_engine.latest_result:
                latest_result = self.au_engine.latest_result
                if 'aus_macro' in latest_result and latest_result['aus_macro']:
                    au_values = latest_result['aus_macro']
                    logger.info(f"[MacroSuggestStrict] Using macro AU data from AUEngine.latest_result: {len(au_values)} AUs")
                elif 'aus' in latest_result and latest_result['aus']:
                    au_values = latest_result['aus']
                    logger.info(f"[MacroSuggestStrict] Using standard AU data from AUEngine.latest_result: {len(au_values)} AUs")
    
            if not au_values or len(au_values) == 0:
                logger.warning("[MacroSuggestStrict] No AU data available. Returning primary result.")
                return primary_result
    
            # 输出原始AU值以便于调试
            logger.debug(f"[MacroSuggestStrict] Original AU values: {au_values}")

            # 标准化AU键，以便与规则中的Lambda函数兼容
            normalized_au_values = {}
            for k, v in au_values.items():
                normalized_key = self.normalize_au_key(k)
                normalized_au_values[normalized_key] = v
                
            au_values = normalized_au_values
            logger.debug(f"[MacroSuggestStrict] Normalized AU values: {au_values}")
    
            au_based_emotion_scores: Dict[Emotion, float] = {}
    
            # 检查是否所有情绪的分数都是0
            all_zeros = True
            
            for emotion, rules in MACRO_EMOTION_AU_MAPPING.items():
                rule_min_intensity = rules.get("min_intensity", 0.0) # Min intensity for required AUs for this rule
                logger.debug(f"[MacroSuggestStrict] Checking {emotion.name}, min_intensity={rule_min_intensity}")
    
                matched_required_aus_intensities = {}
                all_required_met = True
    
                if not rules.get("required"):
                    all_required_met = True # Or handle as a different type of rule if needed.
                                          # Backup implies 'required' AUs are indeed required if the list exists and is non-empty.
                else:
                    for req_au_name in rules["required"]:
                        # Convert AU name (e.g., "AU06") to key used in au_values (e.g., "6")
                        key_to_lookup = self.normalize_au_key(req_au_name)
                        intensity = au_values.get(key_to_lookup, 0.0)
                        logger.debug(f"[MacroSuggestStrict] Required AU {req_au_name} (key={key_to_lookup}): intensity={intensity}, threshold={rule_min_intensity}")
                        
                        if intensity >= rule_min_intensity:
                            matched_required_aus_intensities[key_to_lookup] = intensity
                        else:
                            all_required_met = False
                            break
                
                if not all_required_met:
                    logger.debug(f"[MacroSuggestStrict] {emotion.name}: Not all required AUs met, score=0.0")
                    au_based_emotion_scores[emotion] = 0.0
                    continue
    
                # Consider OPTIONAL AUs
                matched_optional_aus_intensities = {}
                if rules.get("optional"):
                    for opt_au_name in rules["optional"]:
                        key_to_lookup = self.normalize_au_key(opt_au_name)
                        intensity = au_values.get(key_to_lookup, 0.0)
                        optional_threshold = rule_min_intensity * 0.8
                        logger.debug(f"[MacroSuggestStrict] Optional AU {opt_au_name} (key={key_to_lookup}): intensity={intensity}, threshold={optional_threshold}")
                        
                        # Optional AUs contribute if their intensity is >= rule_min_intensity * 0.8 (from backup logic)
                        if intensity >= rule_min_intensity * 0.8:
                            matched_optional_aus_intensities[key_to_lookup] = intensity
                
                all_contributing_aus_intensities = {**matched_required_aus_intensities, **matched_optional_aus_intensities}
                
                if not all_contributing_aus_intensities: # Should only happen if 'required' is empty and no 'optional' met
                    logger.debug(f"[MacroSuggestStrict] {emotion.name}: No contributing AUs found, score=0.0")
                    au_based_emotion_scores[emotion] = 0.0
                    continue
    
                intensity_values = list(all_contributing_aus_intensities.values())
                avg_intensity = sum(intensity_values) / len(intensity_values)
                
                # Base confidence calculation from backup: (avg_intensity / 3.0) * conf_factor
                # The 3.0 is a normalization constant from the backup.
                # Assuming au_values intensities are in a 0-5 (or similar) range as implied by backup thresholds.
                current_emotion_score = (avg_intensity / 3.0) 
                
                conf_factor = rules.get("confidence_factor", 1.0)
                current_emotion_score *= conf_factor
                
                # Apply COHERENCE_CHECK
                coherence_check_config = rules.get("coherence_check")
                if coherence_check_config and callable(coherence_check_config.get("rule")):
                    coherence_result = coherence_check_config["rule"](au_values)
                    logger.debug(f"[MacroSuggestStrict] {emotion.name} coherence check result: {coherence_result}")
                    if not coherence_result:
                        current_emotion_score *= coherence_check_config.get("penalty", 0.8)
                        logger.debug(f"[MacroSuggestStrict] Emotion {emotion.name} failed coherence check, penalty applied.")
                
                # Apply CONFLICTS
                conflicts_config = rules.get("conflicts", {})
                for conflict_au_name, penalty in conflicts_config.items():
                    # Adapt conflict AU name for lookup, e.g., 'AU12' to '12'
                    conflict_key_to_lookup = self.normalize_au_key(conflict_au_name)
                    conflict_threshold = rule_min_intensity * 0.7
                    conflict_intensity = au_values.get(conflict_key_to_lookup, 0.0)
                    
                    # Check if conflicting AU is active enough (intensity >= rule_min_intensity * 0.7)
                    if conflict_intensity >= conflict_threshold:
                        logger.debug(f"[MacroSuggestStrict] {emotion.name} has conflict AU {conflict_au_name} with intensity {conflict_intensity} >= {conflict_threshold}")
                        current_emotion_score *= penalty
                        logger.debug(f"[MacroSuggestStrict] Emotion {emotion.name} has conflict AU {conflict_au_name}, penalty applied.")
                
                # Apply overall minimum confidence and clamp (from backup logic)
                final_score_for_emotion = max(self.minimum_confidence_threshold, min(0.95, current_emotion_score))
                au_based_emotion_scores[emotion] = final_score_for_emotion
                
                if final_score_for_emotion > self.minimum_confidence_threshold:
                    logger.info(f"[MacroSuggestStrict] Detected macro emotion: {emotion.name}, Score: {final_score_for_emotion:.2f}, AvgIntensity: {avg_intensity:.2f}")
                    all_zeros = False

            # 如果没有找到有效的情绪分数，确保不会默认返回中性情绪
            if all_zeros:
                logger.debug("[MacroSuggestStrict] All emotion scores are near zero or below threshold.")
                au_based_emotion_scores[Emotion.NEUTRAL] = 0.8
                logger.info("[MacroSuggestStrict] Defaulting to NEUTRAL with score 0.8")
            
            # 输出所有情绪分数
            for emotion, score in au_based_emotion_scores.items():
                if score > 0.1:  # 只输出有意义的分数
                    logger.debug(f"[MacroSuggestStrict] Emotion score - {emotion.name}: {score:.2f}")

            # --- Decision logic (remains similar, thresholds might need tuning based on new scoring) ---
            final_emotion = primary_result.emotion_type
            final_confidence = primary_result.probability
            suggestion_source = primary_result.source 
    
            if not au_based_emotion_scores or all(s <= self.minimum_confidence_threshold for s in au_based_emotion_scores.values()):
                logger.debug("[MacroSuggestStrict] AU analysis did not yield any significant emotion scores.")
                return primary_result
    
            au_dominant_emotion = max(au_based_emotion_scores, key=au_based_emotion_scores.get, default=None)
            
            if au_dominant_emotion is None or au_based_emotion_scores[au_dominant_emotion] <= self.minimum_confidence_threshold:
                logger.debug("[MacroSuggestStrict] No dominant emotion found from AU analysis or score is too low.")
                return primary_result
                
            au_dominant_confidence = au_based_emotion_scores[au_dominant_emotion]
            logger.info(f"[MacroSuggestStrict] Primary: {primary_result.emotion_type.name} ({primary_result.probability:.2f}), AU Suggests: {au_dominant_emotion.name} ({au_dominant_confidence:.2f})")
    
            override_threshold_diff = self.config_manager.get("emotion_engine.au_override_threshold_diff", 0.2) 
            au_min_confidence_for_override = self.config_manager.get("emotion_engine.au_min_confidence_for_override", 0.5) 
            confirmation_threshold = self.config_manager.get("emotion_engine.au_confirmation_threshold", 0.4) 
    
            if au_dominant_emotion == primary_result.emotion_type:
                if au_dominant_confidence >= confirmation_threshold:
                    final_confidence = max(primary_result.probability, au_dominant_confidence)
                    suggestion_source = f"{primary_result.source}+AU_confirm_strict"
                    logger.info(f"[MacroSuggestStrict] AU confirmed {final_emotion.name}, new confidence: {final_confidence:.2f}")
            else:
                if au_dominant_confidence >= au_min_confidence_for_override and \
                   (au_dominant_confidence > primary_result.probability + override_threshold_diff):
                    final_emotion = au_dominant_emotion
                    final_confidence = au_dominant_confidence
                    suggestion_source = f"AU_override_strict({primary_result.source})"
                    logger.info(f"[MacroSuggestStrict] AU OVERRIDE: from {primary_result.emotion_type.name} to {final_emotion.name} ({final_confidence:.2f})")
            
            final_result_obj = EmotionResult(
                timestamp=primary_result.timestamp, 
                face_id=primary_result.face_id,
                emotion_type=final_emotion,
                probability=final_confidence,
                source=suggestion_source,
                detection_method=primary_result.detection_method, 
                au_data=au_values 
            )
    
            self.event_bus.publish(EventType.AU_MACRO_EMOTION_ANALYZED, final_result_obj, source=self.__class__.__name__)
            logger.info(f"[MacroEmotionAUEngine] Published final emotion (strict): {final_result_obj.emotion_type.name} ({final_result_obj.probability:.2f}) via EventBus.")
    
            return final_result_obj
        except Exception as e:
            logger.error(f"[MacroSuggestStrict] Error processing emotion: {e}")
            logger.error(traceback.format_exc())
            return primary_result

    # 处理事件总线上的主宏观表情结果
    def handle_primary_macro_emotion_event(self, event: Event):
        """处理主宏观情绪引擎发送的事件"""
        try:
            if not self.is_running:
                logger.debug("[MacroEmotionAUEngine] Engine not running, ignoring event")
                return
                
            if event.type == EventType.RAW_MACRO_EMOTION_ANALYZED:
                primary_emotion_result = event.data.get("result")
                frame_data_dict = event.data.get("frame_data")
                
                if primary_emotion_result:
                    logger.info(f"[MacroEmotionAUEngine] Received RAW_MACRO_EMOTION_ANALYZED event for face {primary_emotion_result.face_id}, emotion: {primary_emotion_result.emotion_type.name} ({primary_emotion_result.probability:.2f})")
                    
                    current_frame_data = None
                    if frame_data_dict and isinstance(frame_data_dict, dict) and 'image' in frame_data_dict:
                        try:
                            current_frame_data = FrameData(
                                image=frame_data_dict.get('image'),
                                timestamp=frame_data_dict.get('timestamp', time.time()),
                            )
                            logger.debug("[MacroEmotionAUEngine] Successfully created FrameData from event data")
                        except Exception as e:
                            logger.error(f"Error creating FrameData in MacroEmotionAUEngine: {e}", exc_info=True)
                    
                    # 使用最新的宏观范围AU数据
                    result = self.suggest_alternative_or_confirm(
                        primary_result=primary_emotion_result,
                        frame_data=current_frame_data,
                        current_aus_direct=self.latest_aus_macro if self.latest_aus_macro else None
                    )
                    
                    # 如果结果不同，记录信息
                    if result.emotion_type != primary_emotion_result.emotion_type or abs(result.probability - primary_emotion_result.probability) > 0.05:
                        logger.info(f"[MacroEmotionAUEngine] Altered primary emotion from {primary_emotion_result.emotion_type.name} ({primary_emotion_result.probability:.2f}) to {result.emotion_type.name} ({result.probability:.2f})")
                else:
                    logger.warning("[MacroEmotionAUEngine] Received RAW_MACRO_EMOTION_ANALYZED event without 'result' data.")
        except Exception as e:
            logger.error(f"[MacroEmotionAUEngine] Error handling event: {e}")
            logger.error(traceback.format_exc())