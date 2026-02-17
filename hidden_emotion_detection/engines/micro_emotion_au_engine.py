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
logger = logging.getLogger("MicroEmotionAUEngine")

# 严格遵循备份文件 au_micro_emotion.py.backup 中的微表情AU映射规则
# AU强度值假设与备份文件中的尺度一致
# Lambda函数中的AU键已适配为数值字符串 (例如 '6', '12')
MICRO_EMOTION_AU_MAPPING = {
    Emotion.HAPPY: { # 微弱的喜悦，可能一闪而过
        'required': ['AU06', 'AU12'], # Backup: required: ['AU06', 'AU12']
        'optional': [],             # Backup: optional: []
        'min_intensity': 0.4,       # Backup: min_intensity: 0.4
        'max_intensity': 1.8,       # Backup: max_intensity: 1.8
        'confidence_factor': 1.1    # Backup: confidence_factor: 1.1
    },
    Emotion.SADNESS: { # 微弱的悲伤
        'required': ['AU01', 'AU04'], # Backup: required: ['AU01', 'AU04']
        'optional': ['AU15', 'AU17'],# Backup: optional: ['AU15', 'AU17']
        'min_intensity': 0.5,       # Backup: min_intensity: 0.5
        'max_intensity': 1.8,       # Backup: max_intensity: 1.8
        'confidence_factor': 1.0    # Backup: confidence_factor: 1.0
    },
    Emotion.SURPRISE: { # 微弱的惊讶
        'required': ['AU01', 'AU02'], # Backup: required: ['AU01', 'AU02']
        'optional': ['AU05', 'AU26'],# Backup: optional: ['AU05', 'AU26']
        'min_intensity': 0.35,      # Backup: min_intensity: 0.35
        'max_intensity': 1.8,       # Backup: max_intensity: 1.8
        'confidence_factor': 1.15   # Backup: confidence_factor: 1.15
    },
    Emotion.FEAR: { # 微弱的恐惧/担忧
        'required': [],
        'combinations': [
            {'any': ['AU01', 'AU02', 'AU04', 'AU05', 'AU20'], 'min_count': 2} # Backup rule
        ],
        'min_intensity': 0.35,      # Backup: min_intensity: 0.35
        'max_intensity': 1.5,       # Backup: max_intensity: 1.5
        'confidence_factor': 0.9    # Backup: confidence_factor: 0.9
    },
    Emotion.DISGUST: { # 微弱的厌恶
        'required': ['AU09'],        # Backup: required: ['AU09']
        'optional': ['AU10', 'AU17'],# Backup: optional: ['AU10', 'AU17']
        'min_intensity': 0.4,       # Backup: min_intensity: 0.4
        'max_intensity': 1.7,       # Backup: max_intensity: 1.7
        'confidence_factor': 0.95   # Backup: confidence_factor: 0.95
    },
    Emotion.ANGER: { # 微弱的愤怒/恼怒
        'required': ['AU04', 'AU07'], # Backup: required: ['AU04', 'AU07']
        'optional': ['AU23'],        # Backup: optional: ['AU23']
        'min_intensity': 0.5,       # Backup: min_intensity: 0.5
        'max_intensity': 1.6,       # Backup: max_intensity: 1.6
        'confidence_factor': 0.9    # Backup: confidence_factor: 0.9
    },
    Emotion.CONTEMPT: {
        'required': ['AU12', 'AU14'],# Backup: required: ['AU12', 'AU14']
        'optional': [],
        'min_intensity': 0.5,       # Backup: min_intensity: 0.5
        'max_intensity': 1.5,       # Backup: max_intensity: 1.5
        'confidence_factor': 0.85   # Backup: confidence_factor: 0.85
    },
    # Using Emotion.SUPPRESSED as key, but rules from EmotionType.REPRESSION in backup
    Emotion.SUPPRESSED: { 
        'required': [],
        'combinations': [
            {'any': ['AU04', 'AU07', 'AU17', 'AU23', 'AU24'], 'min_count': 2} # From backup REPRESSION rule
        ],
        'min_intensity': 0.5,       # Backup: min_intensity: 0.5 for REPRESSION
        'max_intensity': 1.6,       # Backup: max_intensity: 1.6 for REPRESSION
        'confidence_factor': 0.8    # Backup: confidence_factor: 0.8 for REPRESSION
    }
}

class MicroEmotionAUEngine:
    """
    微表情AU辅助分析引擎。
    接收主微表情引擎的分析结果，并结合AU数据进行验证、修正或增强。
    """
    
    def __init__(self, 
                 config_manager: ConfigManager,
                 event_bus: EventBus,
                 au_engine: Optional[AUEngine] = None):
        """
        初始化微表情AU辅助引擎
        
        参数:
            config_manager: 配置管理器实例
            event_bus: 事件总线实例
            au_engine: AU引擎实例。如果为None，将创建新实例。
        """
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.is_running = False
        self.frame_count = 0
    
        # 获取或创建AU引擎实例
        if au_engine:
            self.au_engine = au_engine
            logger.info("[MicroEmotionAUEngine] Using provided AUEngine instance.")
        else:
            logger.info("[MicroEmotionAUEngine] No AUEngine provided, creating a new one.")
            models_dir_cfg = self.config_manager.get("au.models_dir", None)
            self.au_engine = AUEngine(
                config_manager=self.config_manager,
                event_bus=self.event_bus,
                models_dir=models_dir_cfg,
                au_threshold=self.config_manager.get("au.threshold", 0.3),
                enable_real_time=self.config_manager.get("au.engine.enable_real_time", True)
            )
            logger.info("[MicroEmotionAUEngine] New AUEngine instance created.")

        # 从配置获取微表情AU分析的最低置信度阈值
        self.minimum_confidence_threshold = self.config_manager.get(
            "micro_emotion_au.minimum_confidence_threshold", 0.02 # From backup
        )
        # 序列分析相关的参数 (从备份文件借鉴)
        self.is_sequence_result = False
        self.sequence_length = 0
        
        # 存储最新的AU数据
        self.latest_au_data = {}
        self.latest_aus_micro = {}  # 专门存储微表情AU数据(0-1.4范围)

        # 添加事件订阅
        if self.event_bus:
            try:
                self.event_bus.subscribe(
                    EventType.RAW_MICRO_EMOTION_ANALYZED, # 微表情引擎消息
                    self.handle_primary_micro_emotion_event
                )
                logger.info("[MicroEmotionAUEngine] Subscribed to RAW_MICRO_EMOTION_ANALYZED event.")
                
                # 额外订阅AU_ANALYZED事件，以便及时获取最新的AU数据
                self.event_bus.subscribe(
                    EventType.AU_ANALYZED,
                    self._on_au_analyzed
                )
                logger.info("[MicroEmotionAUEngine] Also subscribed to AU_ANALYZED event for latest data.")
            except Exception as e:
                logger.error(f"[MicroEmotionAUEngine] Failed to subscribe to events: {e}")
        else:
            logger.warning("[MicroEmotionAUEngine] Event bus not available, cannot subscribe to events.")

        logger.info("MicroEmotionAUEngine initialized.")

    def start(self):
        """
        启动引擎。
        其内部的 AUEngine 需要启动。
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.frame_count = 0
        
        if self.au_engine:
            # MicroEmotionAUEngine 不直接从 AUEngine 接收回调
            self.au_engine.start(result_callback=None) 
            logger.info("[MicroEmotionAUEngine] Internal AUEngine started.")
        
        logger.info("MicroEmotionAUEngine started and ready to process emotion results.")
    
    def stop(self):
        """
        停止引擎
        """
        self.is_running = False
        if self.au_engine:
            self.au_engine.stop()
            logger.info("[MicroEmotionAUEngine] Internal AUEngine stopped.")
        logger.info("MicroEmotionAUEngine stopped.")
        
    def _on_au_analyzed(self, event: Event):
        """处理AU_ANALYZED事件，更新最新的AU数据"""
        try:
            if not self.is_running:
                return
                
            event_data = event.data
            if not isinstance(event_data, dict):
                logger.warning(f"[MicroEmotionAUEngine] Received AU_ANALYZED with unexpected data type: {type(event_data)}")
                return
                
            # 获取AUResult对象
            au_result = event_data.get('result')
            if au_result and hasattr(au_result, 'au_intensities'):
                # 保存标准化AU强度值(0-1范围)
                self.latest_au_data = au_result.au_intensities if au_result.au_intensities else {}
                
                # 保存微表情范围的AU值(0-1.4范围)，用于微表情分析
                if 'aus_micro' in event_data:
                    self.latest_aus_micro = event_data['aus_micro']
                    logger.debug(f"[MicroEmotionAUEngine] Updated micro AU data from event: {len(self.latest_aus_micro)} AUs")
                    
                logger.debug(f"[MicroEmotionAUEngine] Updated AU data from event: {len(self.latest_au_data)} AUs")
        except Exception as e:
            logger.error(f"[MicroEmotionAUEngine] Error processing AU_ANALYZED event: {e}")

    def suggest_alternative_or_confirm(
        self, 
        primary_result: EmotionResult, 
        frame_data: Optional[FrameData] = None, 
        current_aus_direct: Optional[Dict[str, float]] = None,
        is_sequence_input: bool = False, # Added to pass sequence info
        sequence_length_input: int = 0    # Added to pass sequence info
    ) -> EmotionResult:
        """
        根据AU数据，对主微表情引擎的结果提供建议或确认。
        严格遵循备份文件中的AU映射和评分逻辑。
        """
        try:
            if not self.is_running:
                logger.warning("MicroEmotionAUEngine is not running. Returning primary result.")
                return primary_result
                
            logger.info(f"[MicroSuggestStrict] Processing primary micro-emotion: {primary_result.emotion_type.name}, confidence: {primary_result.probability:.2f}")
    
            # Update sequence information for this call
            self.is_sequence_result = is_sequence_input
            self.sequence_length = sequence_length_input
    
            au_values: Optional[Dict[str, float]] = None
            
            # 1. 优先使用直接提供的AU数据
            if current_aus_direct:
                au_values = current_aus_direct
                logger.info(f"[MicroSuggestStrict] Using directly provided AUs: {list(au_values.keys()) if au_values else 'None'}")
            # 2. 尝试从frame_data获取AU数据
            elif frame_data and frame_data.image is not None:
                if self.au_engine:
                    au_engine_output = self.au_engine.process_frame(frame_data.image)
                    if au_engine_output and 'aus_micro' in au_engine_output:
                        # 优先使用微表情范围的AU强度值
                        au_values = au_engine_output['aus_micro']
                        logger.info(f"[MicroSuggestStrict] Using micro AU values from process_frame")
                    elif au_engine_output and 'aus' in au_engine_output:
                        # 退回到原始AU值
                        au_values = au_engine_output['aus']
                        logger.info(f"[MicroSuggestStrict] Using standard AU values from process_frame")
            # 3. 使用最新接收到的AU数据
            elif self.latest_aus_micro:
                au_values = self.latest_aus_micro
                logger.info(f"[MicroSuggestStrict] Using latest micro AU data from event ({len(self.latest_aus_micro)} AUs)")
            # 4. 回退到从AU引擎获取最新结果
            elif self.au_engine and self.au_engine.latest_result:
                latest_result = self.au_engine.latest_result
                if 'aus_micro' in latest_result:
                    au_values = latest_result['aus_micro']
                    logger.info(f"[MicroSuggestStrict] Using micro AU data from AUEngine.latest_result")
                elif 'aus' in latest_result:
                    au_values = latest_result['aus']
                    logger.info(f"[MicroSuggestStrict] Using standard AU data from AUEngine.latest_result")
    
            if not au_values:
                logger.warning("[MicroSuggestStrict] No AU data available. Returning primary result.")
                return primary_result
            
            # 将AU标识从"AU06"格式转换为"6"格式
            def normalize_au_key(key):
                if isinstance(key, str) and key.startswith('AU'):
                    return key[2:].lstrip('0')
                return key
    
            # 标准化AU键，以便与规则中的Lambda函数兼容
            normalized_au_values = {}
            for k, v in au_values.items():
                normalized_key = normalize_au_key(k)
                normalized_au_values[normalized_key] = v
                
            au_values = normalized_au_values
            logger.debug(f"[MicroSuggestStrict] Normalized AU values: {au_values}")
            
            au_based_emotion_scores: Dict[Emotion, float] = {}
    
            for emotion, rule_config in MICRO_EMOTION_AU_MAPPING.items():
                confidence = 0.0
                rule_min_intensity = rule_config.get('min_intensity', 0.05)
                rule_max_intensity = rule_config.get('max_intensity', 2.5) # Default from backup general micro range
    
                # Helper to check AU intensity and convert name
                def get_au_intensity(au_name_with_prefix: str) -> float:
                    key_to_lookup = normalize_au_key(au_name_with_prefix)
                    return au_values.get(key_to_lookup, 0.0)
    
                # 1. Check REQUIRED AUs
                required_aus = rule_config.get('required', [])
                all_required_met_with_intensity = True
                num_required_present = 0
                sum_required_intensity = 0.0
    
                if required_aus:
                    for req_au_name in required_aus:
                        intensity = get_au_intensity(req_au_name)
                        if rule_min_intensity <= intensity <= rule_max_intensity:
                            num_required_present += 1
                            sum_required_intensity += intensity
                        else:
                            all_required_met_with_intensity = False
                            break
                    if not all_required_met_with_intensity:
                        au_based_emotion_scores[emotion] = 0.0
                        continue # Skip to next emotion if required AUs are not met
                
                # 2. Check COMBINATIONS (if required AUs were met or not defined)
                combinations = rule_config.get('combinations', [])
                combination_condition_met = not combinations # True if no combinations defined
                num_combination_aus_active = 0
                sum_combination_intensity = 0.0
    
                if combinations:
                    combination_condition_met = False # Assume false until a combo is met
                    for combo_rule in combinations:
                        any_aus_list = combo_rule.get('any', [])
                        min_count = combo_rule.get('min_count', 1)
                        valid_count_in_combo = 0
                        current_combo_intensity_sum = 0.0
                        
                        for au_name_in_combo in any_aus_list:
                            intensity = get_au_intensity(au_name_in_combo)
                            if rule_min_intensity <= intensity <= rule_max_intensity:
                                valid_count_in_combo += 1
                                current_combo_intensity_sum += intensity
                        
                        if valid_count_in_combo >= min_count:
                            combination_condition_met = True
                            num_combination_aus_active = valid_count_in_combo # Or sum across all met combos if logic changes
                            sum_combination_intensity = current_combo_intensity_sum # Or sum across all met combos
                            break # Found a met combination for this emotion rule
                
                if not combination_condition_met:
                    # If 'required' was also empty, then this rule truly fails
                    # If 'required' was met, but 'combinations' was defined and failed, this rule fails
                    if not required_aus or (required_aus and combinations):
                         au_based_emotion_scores[emotion] = 0.0
                         continue
    
                # At this point, either required AUs are met OR a combination is met (or both if defined that way).
                # Calculate confidence based on backup's _calculate_emotion_confidence_optimized logic
                # This is a simplified adaptation.
                
                # Contribution from required AUs
                if num_required_present > 0:
                    avg_req_intensity = sum_required_intensity / num_required_present
                    confidence += avg_req_intensity * 0.7 # Necessary AU contribution 70% (from backup inspiration)
    
                # Contribution from optional AUs
                optional_aus = rule_config.get('optional', [])
                num_optional_present = 0
                sum_optional_intensity = 0.0
                if optional_aus:
                    for opt_au_name in optional_aus:
                        intensity = get_au_intensity(opt_au_name)
                        if rule_min_intensity <= intensity <= rule_max_intensity:
                            num_optional_present +=1
                            sum_optional_intensity += intensity
                    
                    if num_optional_present > 0:
                        avg_opt_intensity = sum_optional_intensity / num_optional_present
                        opt_ratio = num_optional_present / len(optional_aus) if len(optional_aus) > 0 else 0
                        confidence += avg_opt_intensity * 0.3 * opt_ratio # Optional AU contribution up to 30%
    
                # Contribution from combinations (if they were the primary trigger or additive)
                if num_combination_aus_active > 0 and not required_aus : # If combinations are the main rule part
                    # This part of backup logic: confidence = max(confidence, combo_contribution)
                    # We can make it additive or max, let's try additive scaled by intensity
                    avg_combo_intensity = sum_combination_intensity / num_combination_aus_active
                    combo_base_score = avg_combo_intensity * 0.6 # Combination base score 60%
                    confidence = max(confidence, combo_base_score) # Ensure combo can set a base if others are low
                
                # Apply emotion-specific confidence factor
                confidence *= rule_config.get('confidence_factor', 1.0)
    
                # Sequence data enhancement (from backup)
                if self.is_sequence_result and self.sequence_length >= 3:
                    sequence_bonus = min(0.3, self.sequence_length * 0.03) # Max 30% bonus
                    confidence *= (1.0 + sequence_bonus)
                
                final_score_for_emotion = max(self.minimum_confidence_threshold, min(1.0, confidence)) # Clamp 0-1
                au_based_emotion_scores[emotion] = final_score_for_emotion
                if final_score_for_emotion > self.minimum_confidence_threshold:
                    logger.debug(f"[MicroSuggestStrict] Emotion: {emotion.name}, Score: {final_score_for_emotion:.3f}")
                    
            # --- Decision logic (similar to Macro, thresholds from config might differ for micro) ---
            final_emotion = primary_result.emotion_type
            final_confidence = primary_result.probability
            suggestion_source = primary_result.source
    
            if not au_based_emotion_scores or all(s == 0 for s in au_based_emotion_scores.values()):
                logger.debug("[MicroSuggestStrict] AU analysis did not yield any significant micro-emotion scores.")
                return primary_result
    
            au_dominant_emotion = max(au_based_emotion_scores, key=au_based_emotion_scores.get, default=None)
            if au_dominant_emotion is None or au_based_emotion_scores[au_dominant_emotion] == 0:
                logger.debug("[MicroSuggestStrict] No dominant micro-emotion found from AU or score is zero.")
                return primary_result
                
            au_dominant_confidence = au_based_emotion_scores[au_dominant_emotion]
            logger.info(f"[MicroSuggestStrict] Primary Micro: {primary_result.emotion_type.name} ({primary_result.probability:.2f}), AU Suggests Micro: {au_dominant_emotion.name} ({au_dominant_confidence:.2f})")
    
            override_threshold_diff = self.config_manager.get("micro_emotion.au_override_threshold_diff", 0.25)
            au_min_confidence_for_override = self.config_manager.get("micro_emotion.au_min_confidence_for_override", 0.4)
            confirmation_threshold = self.config_manager.get("micro_emotion.au_confirmation_threshold", 0.3)
    
            if au_dominant_emotion == primary_result.emotion_type:
                if au_dominant_confidence >= confirmation_threshold:
                    final_confidence = max(primary_result.probability, au_dominant_confidence)
                    suggestion_source = f"{primary_result.source}+AU_micro_confirm_strict"
                    logger.info(f"[MicroSuggestStrict] AU confirmed micro {final_emotion.name}, new confidence: {final_confidence:.2f}")
            elif au_dominant_confidence >= au_min_confidence_for_override and \
                 (au_dominant_confidence > primary_result.probability + override_threshold_diff):
                final_emotion = au_dominant_emotion
                final_confidence = au_dominant_confidence
                suggestion_source = f"AU_micro_override_strict({primary_result.source})"
                logger.info(f"[MicroSuggestStrict] AU OVERRIDE micro: from {primary_result.emotion_type.name} to {final_emotion.name} ({final_confidence:.2f})")
            
            final_result_obj = EmotionResult(
                timestamp=primary_result.timestamp,
                face_id=primary_result.face_id,
                emotion_type=final_emotion,
                probability=final_confidence,
                source=suggestion_source,
                detection_method=primary_result.detection_method,
                au_data=au_values,
                is_micro_expression=True # Mark as micro-expression result
            )
            
            self.event_bus.publish(EventType.AU_MICRO_EMOTION_ANALYZED, final_result_obj, source=self.__class__.__name__)
            logger.info(f"[MicroEmotionAUEngine] Published final micro-emotion (strict): {final_result_obj.emotion_type.name} ({final_result_obj.probability:.2f}) via EventBus.")
    
            return final_result_obj
        except Exception as e:
            logger.error(f"[MicroSuggestStrict] Error processing emotion: {e}")
            logger.error(traceback.format_exc())
            return primary_result

    def handle_primary_micro_emotion_event(self, event: Event):
        """处理主微表情引擎发送的事件"""
        try:
            if not self.is_running:
                logger.debug("[MicroEmotionAUEngine] Engine not running, ignoring event")
                return
                
            if event.type == EventType.RAW_MICRO_EMOTION_ANALYZED:
                primary_emotion_result = event.data.get("result")
                frame_data_dict = event.data.get("frame_data")
                
                # 获取序列信息
                is_sequence = event.data.get("is_sequence", False)
                sequence_length = event.data.get("frames_count", 0)
    
                if primary_emotion_result:
                    logger.info(f"[MicroEmotionAUEngine] Received RAW_MICRO_EMOTION_ANALYZED event for face {primary_emotion_result.face_id}, emotion: {primary_emotion_result.emotion_type.name} ({primary_emotion_result.probability:.2f})")
                    
                    current_frame_data = None
                    if frame_data_dict and isinstance(frame_data_dict, dict) and frame_data_dict.get('image') is not None:
                        try:
                            current_frame_data = FrameData(
                                image=frame_data_dict.get('image'),
                                timestamp=frame_data_dict.get('timestamp', time.time()),
                            )
                            logger.debug("[MicroEmotionAUEngine] Successfully created FrameData from event data")
                        except Exception as e:
                            logger.error(f"Error creating FrameData in MicroEmotionAUEngine: {e}", exc_info=True)
                    
                    # 使用最新的微表情范围AU数据，并传递序列信息
                    result = self.suggest_alternative_or_confirm(
                        primary_result=primary_emotion_result,
                        frame_data=current_frame_data, 
                        current_aus_direct=self.latest_aus_micro if self.latest_aus_micro else None,
                        is_sequence_input=is_sequence,
                        sequence_length_input=sequence_length
                    )
                    
                    # 如果结果不同，记录信息
                    if result.emotion_type != primary_emotion_result.emotion_type or abs(result.probability - primary_emotion_result.probability) > 0.05:
                        logger.info(f"[MicroEmotionAUEngine] Altered primary micro-emotion from {primary_emotion_result.emotion_type.name} ({primary_emotion_result.probability:.2f}) to {result.emotion_type.name} ({result.probability:.2f})")
                else:
                    logger.warning("[MicroEmotionAUEngine] Received RAW_MICRO_EMOTION_ANALYZED event without 'result' data.")
        except Exception as e:
            logger.error(f"[MicroEmotionAUEngine] Error handling event: {e}")
            logger.error(traceback.format_exc())

# 类的结尾