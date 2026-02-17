#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人脸检测引擎
检测视频帧中的所有人脸，并提取面部特征点
"""

import os
import dlib
import cv2
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 导入核心数据类型
from hidden_emotion_detection.core.data_types import FaceDetection, FaceBox, Landmarks, FacePose, EventType
from hidden_emotion_detection.core.event_bus import EventBus
from hidden_emotion_detection.config.config_manager import ConfigManager # 导入配置管理器

# 导入姿态估计引擎
from hidden_emotion_detection.engines.pose_estimator import PoseEstimationEngine

# 配置日志
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceDetectionEngine")

# ---- 添加辅助函数 ----
def shape_to_np(shape, dtype="int"):
    # 创建 (68, 2) 形状的零数组
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历所有面部特征点
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # 返回特征点坐标
    return coords
# ---- 添加辅助函数 ----

class FaceDetectionEngine:
    """人脸检测引擎，支持多种方法并根据配置选择"""
    
    def __init__(self, config: ConfigManager): # 接收 ConfigManager 实例
        """
        初始化人脸检测引擎
        
        Args:
            config: 配置管理器实例
        """
        all_configs = config.get_all() # 获取所有配置
        self.face_config = all_configs.get('face', {}) # 获取人脸检测配置字典
        self.system_config = all_configs.get('system', {}) # 获取系统配置字典
        models_dir = self.system_config.get('models_dir', './enhance_hidden/models')
        
        logger.info("初始化人脸检测引擎...")
        
        # 从获取的字典中读取配置
        self.detection_method = self.face_config.get('detection_method', 'dlib').lower()
        self.use_landmarks = self.face_config.get('use_landmarks', True)
        self.landmark_method = self.face_config.get('landmark_method', 'dlib').lower()
        # --- 添加 use_pose 读取 ---
        # 姿态估计依赖于关键点，所以如果 use_landmarks 为 False，则 use_pose 也应为 False
        self.use_pose = self.face_config.get('use_pose', True) and self.use_landmarks
        # --- 添加 use_pose 读取 ---
        
        self.detector = None
        self.predictor = None
        self.cascade = None
        self.mp_face_detection = None
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ---- 添加 paused 属性初始化 ----
        self.paused = False 
        # ---- 添加 paused 属性初始化 ----
        
        # 根据配置初始化检测器
        if self.detection_method == 'dlib':
            logger.info("使用 dlib HOG 检测器")
            self.detector = dlib.get_frontal_face_detector()
        elif self.detection_method == 'haar':
            cascade_path = self.face_config.get('haar_cascade_path',
                                            os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))
            if os.path.exists(cascade_path):
                logger.info(f"使用 Haar Cascade 检测器: {cascade_path}")
                self.cascade = cv2.CascadeClassifier(cascade_path)
                self.detector = self.cascade # 将加载的 cascade 赋值给 detector
            else:
                logger.error(f"Haar Cascade 文件未找到: {cascade_path}")
                self.cascade = None
                self.detector = None # 明确设为 None
        elif self.detection_method == 'mediapipe':
            logger.info("使用 MediaPipe Face Detection")
            try:
                # 提供一个默认的模型路径（如果未在配置中指定）
                default_mp_model = 'blaze_face_short_range.tflite'
                mp_model_path = os.path.join(models_dir, self.face_config.get('mediapipe_model', default_mp_model))
                
                if not os.path.exists(mp_model_path):
                     logger.error(f"MediaPipe 模型文件未找到: {mp_model_path}. 请确保模型文件存在于 models_dir 中。")
                     raise FileNotFoundError(f"MediaPipe model not found at {mp_model_path}")
                     
                logger.info(f"加载 MediaPipe 模型: {mp_model_path}")
                # 创建配置选项
                base_options = python.BaseOptions(model_asset_path=mp_model_path)
                options = vision.FaceDetectorOptions(base_options=base_options,
                                                    min_detection_confidence=self.face_config.get('confidence_threshold', 0.5))
                self.detector = vision.FaceDetector.create_from_options(options)
            except Exception as e:
                 logger.error(f"初始化 MediaPipe Face Detection 失败: {e}", exc_info=True)
                 logger.warning("将回退到 dlib HOG 检测器。")
                 self.detector = dlib.get_frontal_face_detector()
                 self.detection_method = 'dlib'
        else:
            logger.warning(f"不支持的人脸检测方法: {self.detection_method}. 将使用 dlib HOG。")
            self.detector = dlib.get_frontal_face_detector()
            self.detection_method = 'dlib' # 更新实际使用的方法
            
        # 根据配置初始化关键点检测器
        if self.use_landmarks:
            if self.landmark_method == 'dlib':
                # 从 face_config 获取 landmark_model 文件名
                landmark_model_filename = self.face_config.get('landmark_model') 
                predictor_path = None
                if landmark_model_filename:
                     # 拼接完整路径
                     predictor_path = os.path.join(models_dir, landmark_model_filename)
                
                # 尝试默认路径（如果配置中没有或拼接后路径不存在）
                if not predictor_path or not os.path.exists(predictor_path):
                     default_predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
                     if os.path.exists(default_predictor_path):
                          predictor_path = default_predictor_path
                     else:
                          predictor_path = None # 最终还是没找到

                if predictor_path:
                    logger.info(f"使用 dlib 关键点检测器: {predictor_path}")
                    self.predictor = dlib.shape_predictor(predictor_path)
                else: # 如果 predictor 未成功加载
                    logger.error(f"dlib 关键点模型文件未能找到或加载。关键点检测将被禁用。")
                    self.use_landmarks = False # 禁用关键点
                    self.use_pose = False # 同时禁用姿态估计
            else:
                 logger.warning(f"不支持的关键点检测方法: {self.landmark_method}. 关键点检测将被禁用。")
                 self.use_landmarks = False # 禁用关键点
                 self.use_pose = False # 同时禁用姿态估计
        else:
            logger.info("关键点检测已禁用")
        
        # 追踪相关变量
        self.last_face_id = 0
        self.tracked_faces = {}  # 字典存储追踪的人脸 {face_id: {last_box:, last_seen:}}
        self.max_tracking_age = self.face_config.get('tracking_age', 30) # 从配置获取
        
        # 引擎状态
        self.is_running = False
        
        # 获取事件总线实例
        self.event_bus = EventBus()
        
        # 创建姿态估计引擎实例 (仅在需要时)
        self.pose_engine = None
        if self.use_landmarks and self.use_pose:
            logger.info("初始化姿态估计引擎...")
            self.pose_engine = PoseEstimationEngine()
        elif self.use_landmarks:
            logger.info("姿态估计已禁用 (根据配置 use_pose=false)")
        else:
            logger.info("姿态估计已禁用，因为关键点检测被禁用。")
        
        logger.info("人脸检测引擎初始化完成")

    def _detect_with_mediapipe(self, frame_rgb: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """使用 MediaPipe 检测人脸。期望输入是 RGB 帧。"""
        detections_out = []
        confidences_out = []
        
        if not self.detector: # MediaPipe 使用 self.detector
            logger.error("MediaPipe detector not initialized.")
            return detections_out, confidences_out

        try:
            # 将 NumPy 数组转换为 MediaPipe 的 Image 对象
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # 执行检测
            detection_result = self.detector.detect(mp_image)

            if detection_result.detections:
                # image_height, image_width, _ = frame_rgb.shape # 不再需要，因为返回的是绝对坐标
                for detection in detection_result.detections:
                    # bounding_box 包含 origin_x, origin_y, width, height (已经是绝对像素坐标)
                    bbox = detection.bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    
                    detections_out.append((int(x), int(y), int(w), int(h)))
                    
                    # 获取置信度
                    confidence = 0.0 # Default confidence
                    if detection.categories:
                         try:
                              confidence = detection.categories[0].score
                         except (IndexError, AttributeError):
                              logger.warning("无法从 MediaPipe detection category 获取置信度分数。")
                    confidences_out.append(confidence)

                    # 还可以获取关键点 (如果需要的话)
                    # keypoints = detection.keypoints # 包含 6 个关键点 (左右眼, 鼻尖, 嘴中心, 左右耳屏)
                    # 每个 keypoint 有 x, y (归一化), keypoint_name, score
            
        except Exception as e:
            logger.error(f"MediaPipe 人脸检测出错: {e}", exc_info=True)

        return detections_out, confidences_out
        
    def _detect_with_selected_method(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """根据配置选择并执行人脸检测方法。"""
        detections = []
        confidences = [] 
        logger.debug(f"[_detect_with_selected_method] Called with method: {self.detection_method}")

        if self.detection_method == 'dlib' and self.detector:
            # Dlib 需要灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            logger.debug(f"[_detect_with_selected_method] Running dlib detector...")
            # 注意：dlib 检测器可能没有置信度，或者置信度计算方式不同
            dlib_faces, dlib_scores, _ = self.detector.run(gray_frame, self.face_config.get('upsample_times', 1), -1) # 使用 run 获取分数
            logger.debug(f"[_detect_with_selected_method] dlib detector returned {len(dlib_faces)} faces.")
            min_confidence = self.face_config.get('confidence_threshold', -1) # Dlib HOG 分数通常>0, 0.5可能太高
            for dlib_face, score in zip(dlib_faces, dlib_scores):
                 if score >= min_confidence: # 应用置信度阈值
                     detections.append((dlib_face.left(), dlib_face.top(), dlib_face.width(), dlib_face.height())) 
                 confidences.append(score) # 记录 dlib 分数
                 # else: logger.debug(f"Dlib face skipped due to low score: {score:.2f}")
        elif self.detection_method == 'haar' and self.detector:
            # Haar 需要灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            logger.debug(f"[_detect_with_selected_method] Running Haar detector...")
            min_face_size = self.face_config.get('min_face_size', 30)
            # detectMultiScale3 返回 rectangles, rejectLevels, levelWeights
            haar_faces, _, levelWeights = self.detector.detectMultiScale3(
                gray_frame,
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(min_face_size, min_face_size),
                outputRejectLevels=True # 获取置信度分数
            )
            logger.debug(f"[_detect_with_selected_method] Haar detector returned {len(haar_faces)} faces.")
            min_confidence = self.face_config.get('confidence_threshold', 0.5) # Haar levelWeights 解释可能不同
            if len(haar_faces) > 0:
                for i, (x, y, w, h) in enumerate(haar_faces):
                    score = levelWeights[i][0] # 获取置信度
                    # if score >= min_confidence: # Haar 的置信度意义可能不同，需要测试
                    detections.append((x, y, w, h))
                    confidences.append(score) # 记录 Haar 分数
                    # else: logger.debug(f"Haar face skipped due to low score: {score:.2f}")
        elif self.detection_method == 'mediapipe':
            # MediaPipe 需要 RGB 图
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame # Convert BGR to RGB
            logger.debug(f"[_detect_with_selected_method] Running MediaPipe detector...")
            detections_mp, confidences_mp = self._detect_with_mediapipe(frame_rgb) # 调用新方法
             # MediaPipe 内部已应用置信度阈值
            detections = detections_mp
            confidences = confidences_mp
            logger.debug(f"[_detect_with_selected_method] MediaPipe detector returned {len(detections)} faces.")
        else:
             logger.warning(f"[_detect_with_selected_method] 检测器未正确初始化或配置方法 '{self.detection_method}' 不支持内部检测调用。")

        return detections, confidences
        
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        检测帧中的人脸，并进行关键点和姿态估计（如果启用）。
        返回包含所有信息的 FaceDetection 对象列表。
        """
        # ---- 添加入口日志 ----
        logger.debug("[detect_faces] Function entered.")
        # ---- 添加入口日志 ----

        # --- 计时开始 ---
        total_start_time = time.perf_counter()

        # 检查是否暂停
        if self.paused:
            logger.debug("[detect_faces] Engine paused, returning empty list.") # 添加日志
            return [] # 如果暂停，不进行任何处理

        # 检查输入帧
        if frame is None:
            logger.warning("[detect_faces] Received None frame, returning empty list.") # 修改日志
            return []
        
        # 使用选择的检测方法检测人脸
        detection_start_time = time.perf_counter()
        detections, confidence_list = self._detect_with_selected_method(frame)
        detection_time = time.perf_counter() - detection_start_time

        face_results: List[FaceDetection] = []
        landmarks_time_total = 0.0
        pose_time_total = 0.0
        crop_time_total = 0.0

        # 遍历检测结果
        for i, (x, y, width, height) in enumerate(detections):
            confidence = confidence_list[i] if i < len(confidence_list) else 0.0 

            # --- 根据检测方法创建 FaceBox ---
            if self.detection_method == 'haar':
                x1, y1, x2, y2 = float(x), float(y), float(x + width), float(y + height)
            elif self.detection_method == 'dlib':
                x1, y1, x2, y2 = float(x), float(y), float(x + width), float(y + height)
            else: # 默认处理
                x1, y1, x2, y2 = float(x), float(y), float(x + width), float(y + height)

            face_box = FaceBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # 初始化关键点和姿态
            landmarks: Optional[Landmarks] = None
            pose: Optional[FacePose] = None
            face_chip: Optional[np.ndarray] = None

            # 预测关键点 (如果启用)
            landmarks_start_time = time.perf_counter()
            if self.use_landmarks and self.predictor:
                try:
                    # dlib 需要 dlib.rectangle 对象
                    # 使用检测到的整数坐标创建 dlib_rect
                    dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                    shape = self.predictor(frame, dlib_rect) # 在原始彩色帧上预测
                    points = [(shape.part(j).x, shape.part(j).y) for j in range(shape.num_parts)]
                    landmarks = Landmarks(points=points, confidence=1.0) # dlib 不直接提供置信度
                except Exception as e:
                    # 打印更详细的错误信息
                    logger.error(f"dlib 关键点预测失败 for box ({x},{y},{width},{height}): {e}", exc_info=True) 
                    landmarks = None
            landmarks_time = time.perf_counter() - landmarks_start_time
            landmarks_time_total += landmarks_time

            # 预测关键点后，立即分配 Face ID
            face_id = self._assign_face_id(face_box)

            # 估计姿态 (如果启用且成功预测关键点)
            pose_start_time = time.perf_counter()
            if self.use_pose and self.pose_engine and landmarks:
                try:
                    # --- 日志现在可以安全使用 face_id --- 
                    logger.debug(f"[FaceDetectionEngine] Calling pose estimation for face ID {face_id}. use_pose={self.use_pose}, pose_engine_valid={self.pose_engine is not None}, landmarks_valid={landmarks is not None}, num_landmarks={len(landmarks.points) if landmarks else 'N/A'}")
                    # --- 日志现在可以安全使用 face_id ---
                    
                    # --- 修复：创建临时的 FaceDetection 对象传递给 estimate_pose ---
                    temp_face_for_pose = FaceDetection(
                        face_id=face_id, 
                        face_box=face_box, # Pass the box too, might be useful later
                        confidence=confidence, # Pass confidence
                        landmarks=landmarks, # Pass the actual landmarks object
                        pose=None, # Pose is not yet calculated
                        face_chip=None # Chip not needed for pose estimation itself
                    )
                    estimated_pose = self.pose_engine.estimate_pose(frame, temp_face_for_pose) # <--- 传递对象
                    # --- 修复结束 ---

                    if estimated_pose:
                        pose = estimated_pose # 只有成功才赋值给 pose
                    else:
                        logger.debug(f"[FaceDetectionEngine] Pose estimation returned None for face ID {face_id}.") # 添加估计失败日志
                except Exception as e:
                    logger.error(f"姿态估计失败: {e}", exc_info=True)
                    pose = None
            # --- 添加未调用姿态估计的日志 ---
            elif not self.use_pose:
                 logger.debug(f"[FaceDetectionEngine] Pose estimation skipped for face ID {face_id}: use_pose is False.")
            elif not self.pose_engine:
                 logger.debug(f"[FaceDetectionEngine] Pose estimation skipped for face ID {face_id}: pose_engine is None.")
            elif not landmarks:
                 logger.debug(f"[FaceDetectionEngine] Pose estimation skipped for face ID {face_id}: landmarks are None.")
            # --- 添加未调用姿态估计的日志 ---

                 pose_time = time.perf_counter() - pose_start_time
                 pose_time_total += pose_time

            # 裁剪人脸图像
            crop_start_time = time.perf_counter()
            try:
                 face_chip = self._crop_face(frame, face_box)
                 # 检查裁剪结果
                 if face_chip is None or face_chip.size == 0:
                      logger.warning(f"裁剪人脸图像失败或结果为空 for box {face_box}. face_chip 将为 None。")
                      face_chip = None # 确保是 None
            except Exception as e:
                 logger.error(f"裁剪人脸图像时出错 for box {face_box}: {e}", exc_info=True)
                 face_chip = None
            crop_time = time.perf_counter() - crop_start_time
            crop_time_total += crop_time
            
            # 创建 FaceDetection 对象
            face_detection = FaceDetection(
                face_id=face_id,
                face_box=face_box,
                confidence=confidence,
                landmarks=landmarks,
                pose=pose,
                face_chip=face_chip
            )
            face_results.append(face_detection)

        # 更新追踪信息
        current_ids = {face.face_id for face in face_results}
        self._update_tracking(current_ids)
        
        # --- 计时结束和日志记录 ---
        total_time = time.perf_counter() - total_start_time
        logger.debug(f"[TIMER] detect_faces Total: {total_time*1000:.2f}ms | Detection: {detection_time*1000:.2f}ms | Landmarks: {landmarks_time_total*1000:.2f}ms | Pose: {pose_time_total*1000:.2f}ms | Crop: {crop_time_total*1000:.2f}ms")

        # ---- 添加出口日志 ----
        logger.debug(f"[detect_faces] Reached end of function. Number of faces found: {len(face_results)}")
        # ---- 添加出口日志 ----

        return face_results
    
    def _assign_face_id(self, face_box: FaceBox) -> int:
        """
        为检测到的人脸分配一个 ID，尝试匹配现有追踪的人脸。
        简单实现：基于 IoU 匹配。可以替换为更复杂的追踪算法。
        """
        best_match_id = -1
        max_iou = 0.4 # IoU 阈值，可调整

        # 计算当前框的中心点
        current_center_x = (face_box.x1 + face_box.x2) / 2
        current_center_y = (face_box.y1 + face_box.y2) / 2

        for face_id, tracked_data in self.tracked_faces.items():
            last_box = tracked_data['last_box']
            
            # 计算 IoU (交并比)
            xA = max(face_box.x1, last_box.x1)
            yA = max(face_box.y1, last_box.y1)
            xB = min(face_box.x2, last_box.x2)
            yB = min(face_box.y2, last_box.y2)
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (face_box.x2 - face_box.x1) * (face_box.y2 - face_box.y1)
            boxBArea = (last_box.x2 - last_box.x1) * (last_box.y2 - last_box.y1)
            
            iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            
            if iou > max_iou:
                    max_iou = iou
                    best_match_id = face_id
                    
        if best_match_id != -1:
            # 匹配成功，更新追踪信息
            self.tracked_faces[best_match_id]['last_box'] = face_box
            self.tracked_faces[best_match_id]['last_seen'] = 0 # 重置年龄
            return best_match_id
        else:
            # 没有匹配，分配新 ID
            self.last_face_id += 1
            self.tracked_faces[self.last_face_id] = {'last_box': face_box, 'last_seen': 0}
            return self.last_face_id
    
    def _update_tracking(self, current_ids: set):
        """更新追踪列表，增加未出现人脸的年龄，移除过老的人脸。"""
        faces_to_remove = []
        for face_id in list(self.tracked_faces.keys()): # 使用 list 包装以允许在迭代时修改字典
            if face_id not in current_ids:
                self.tracked_faces[face_id]['last_seen'] += 1
                if self.tracked_faces[face_id]['last_seen'] > self.max_tracking_age:
                    faces_to_remove.append(face_id)
            # else: # 如果人脸在当前帧出现，则其 last_seen 已在 _assign_face_id 中重置为 0
                 # pass

        for face_id in faces_to_remove:
                logger.info(f"移除追踪超时的人脸 ID: {face_id}")
                del self.tracked_faces[face_id]
        
    def _estimate_pose(self, frame, landmarks: List[Tuple[int, int]], face_box: FaceBox) -> Optional[FacePose]:
        """
        从面部特征点估计姿态
        
        Args:
            frame: 输入图像帧
            landmarks: 面部特征点
            face_box: 人脸边界框
            
        Returns:
            Optional[FacePose]: 面部姿态或None
        """
        # deprecated or handled by PoseEstimationEngine
        # if self.pose_engine:
        #     return self.pose_engine.estimate_pose(frame, landmarks)
        # return None
        pass # 这个方法不再直接使用
    
    def start(self):
        """启动引擎"""
        self.is_running = True
        logger.info("人脸检测引擎已启动")
        self.event_bus.publish(EventType.ENGINE_STARTED, {"engine": "face"})
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        self.tracked_faces = {} # 清空追踪信息
        self.last_face_id = 0
        logger.info("人脸检测引擎已停止") 
        self.event_bus.publish(EventType.ENGINE_STOPPED, {"engine": "face"})
    
    def _crop_face(self, frame: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        """根据 FaceBox 裁剪人脸区域，并添加一些边距。"""
        if frame is None or face_box is None:
            return None
        
        # --- 使用整数坐标进行裁剪 --- 
        x1, y1, x2, y2 = map(int, face_box.to_tlbr())
        
        # 检查坐标有效性
        if x1 >= x2 or y1 >= y2:
             logger.warning(f"无效的裁剪坐标: ({x1},{y1}) -> ({x2},{y2})")
             return None
        
        # 添加边距（例如，15% 的宽度/高度）
        border_scale = 0.15 
        width = x2 - x1
        height = y2 - y1
        border_x = int(width * border_scale)
        border_y = int(height * border_scale)
            
        # 计算带边距的裁剪坐标，并确保在图像范围内
        frame_h, frame_w = frame.shape[:2]
        crop_x1 = max(0, x1 - border_x)
        crop_y1 = max(0, y1 - border_y)
        crop_x2 = min(frame_w, x2 + border_x)
        crop_y2 = min(frame_h, y2 + border_y)
        
        # 再次检查裁剪坐标有效性
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
             logger.warning(f"添加边距后裁剪坐标无效: ({crop_x1},{crop_y1}) -> ({crop_x2},{crop_y2})")
             # 尝试返回不带边距的裁剪？或者直接返回 None？返回 None 更安全
             return None
            
        try:
            # 执行裁剪
            face_chip = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            logger.debug(f"Cropped face chip shape: {face_chip.shape if face_chip is not None else 'None'}") # 添加日志
            return face_chip
        except Exception as e:
            logger.error(f"裁剪人脸时出错: {e}")
            return None

    def pause(self):
        """暂停引擎处理"""
        self.paused = True
        logger.info("人脸检测引擎已暂停")
        self.event_bus.publish(EventType.ENGINE_PAUSED, {"engine": "face"})

    def resume(self):
        """恢复引擎处理"""
        self.paused = False
        logger.info("人脸检测引擎已恢复")
        self.event_bus.publish(EventType.ENGINE_RESUMED, {"engine": "face"})

