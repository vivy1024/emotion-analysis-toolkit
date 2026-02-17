#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 人脸提取面板
用于显示检测到的人脸和关键点
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging # 添加 logging

from .base_panel import BasePanel
from ..core.data_types import FaceDetection, EventType, Event
from ..core.event_bus import EventBus

logger = logging.getLogger(__name__) # 获取 logger

class FacePanel(BasePanel):
    """人脸提取面板，显示人脸检测结果和关键点"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化人脸提取面板"""
        super().__init__("姿态与人脸") # 更新标题
        self.event_bus = event_bus # 存储传入的 event_bus
        
        # ---- 添加颜色定义 ----
        self.text_color = (200, 200, 200) # 与 AUIntensityPanel 保持一致
        self.highlight_color = (100, 200, 250) # 可选，如果需要高亮
        # ---- 添加颜色定义 ----

        # Assume zoom is always on for simplicity matching OpenFace appearance panel
        self.zoom_face = True
        
        # Current data
        self.current_face: Optional[FaceDetection] = None # 类型改为 FaceDetection
        self.face_image_display = None # Store the processed image for display
        self.landmarks_to_draw = None # Store landmarks processed for display
        self.face_found = False
        self.face_track_duration = 0.0
        self.pose_orientation = {"Turn": 0.0, "Up/down": 0.0, "Tilt": 0.0}
        self.pose_position = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.current_frame: Optional[np.ndarray] = None # 添加用于存储当前帧的属性
        
        # 移除统计属性，因为现在不接收 FrameResult
        # self.face_size = 0.0
        # self.face_detection_rate = 0.0
        # self.total_frames = 0
        # self.face_detected_frames = 0
        # self.last_detection_times = [] 
        # self.detection_rate_window = 10
        
        # ---- 修改事件订阅 ----
        if self.event_bus: # 检查传入的 event_bus 是否存在
            try:
                self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
                logger.info("FacePanel subscribed to FACE_DETECTED event.")
            except Exception as e:
                logger.error(f"Failed to subscribe FacePanel to event: {e}", exc_info=True)
        else:
            logger.warning("No event bus provided to FacePanel, cannot subscribe to events.")
        # ---- 修改事件订阅 ----

    def _on_face_detected(self, event: Event):
        """ 处理人脸检测事件 """
        # 使用 event.data 访问数据
        event_data = event.data 
        if not isinstance(event_data, dict):
             logger.error(f"FacePanel received unexpected data type in event: {type(event_data)}")
             return
             
        logger.debug(f"FacePanel received FACE_DETECTED event data keys: {list(event_data.keys())}")
        
        # ---- 保存原始帧 ----
        # self.current_frame = event_data.get('frame') # 不再在这里存储帧
        # ---- 保存原始帧 ----
        
        # ---- 修改人脸数据获取逻辑 ----
        detected_face: Optional[FaceDetection] = event_data.get('face') 
        # ---- 修改人脸数据获取逻辑 ----

        # if not face_detections: # 旧逻辑
        if detected_face is None:
            # 没有检测到人脸
            if self.face_found: # 只有在状态改变时才更新
                self.face_found = False
                self.current_face = None
                self.face_image_display = None 
                self.landmarks_to_draw = None 
                self.pose_orientation = {"Turn": 0.0, "Up/down": 0.0, "Tilt": 0.0}
                self.pose_position = {"X": 0.0, "Y": 0.0, "Z": 0.0}
                logger.debug("FacePanel updated state: face_found = False")
        else:
            # 检测到人脸
            detected_before = self.face_found
            self.face_found = True
            # self.current_face = face_detections[0] # 旧逻辑
            self.current_face = detected_face # <--- 使用获取到的单个人脸对象
            if not detected_before:
                 logger.debug("FacePanel updated state: face_found = True")

            # 更新姿态信息 (如果存在)
            if self.current_face and self.current_face.pose:
                pose = self.current_face.pose
                self.pose_orientation["Turn"] = pose.yaw
                self.pose_orientation["Up/down"] = pose.pitch
                self.pose_orientation["Tilt"] = pose.roll
                if pose.x_pos is not None:
                    self.pose_position["X"] = pose.x_pos
                    self.pose_position["Y"] = pose.y_pos
                    self.pose_position["Z"] = pose.z_pos
            else:
                self.pose_orientation = {"Turn": 0.0, "Up/down": 0.0, "Tilt": 0.0}
                self.pose_position = {"X": 0.0, "Y": 0.0, "Z": 0.0}

            self.face_image_display = None 
            self.landmarks_to_draw = None

    def _prepare_face_image_for_display(self, original_frame: Optional[np.ndarray]):
        """ Prepare the cropped/zoomed face image. Return display dims and transform params. """
        # Reset state
        self.face_image_display = None
        
        # Default return values
        display_width, display_height = 0, 0
        transform_scale_x, transform_scale_y = 1.0, 1.0
        transform_offset_x, transform_offset_y = 0.0, 0.0

        if not self.face_found or self.current_face is None:
            # print("[FacePanel Prep] No face found or current_face is None") # Debug
            return 0, 0, 1.0, 1.0, 0.0, 0.0 # 返回默认值

        # ---- 添加检查 original_frame 是否为 None ----
        if original_frame is None:
            logger.debug("[FacePanel Prep] original_frame is None, cannot prepare image.") # 修改为 debug
            self.face_image_display = None
            self.landmarks_to_draw = None
            return 0, 0, 1.0, 1.0, 0.0, 0.0
        # ---- 添加检查 original_frame 是否为 None ----
        logger.debug(f"[FacePanel Prep] Original frame shape: {original_frame.shape}") # 添加日志

        max_display_height = 150 
        max_display_width = 150 

        face_img_to_resize = None
        ref_crop_width, ref_crop_height = 0, 0 

        # --- Try using face_chip first ---
        if hasattr(self.current_face, 'face_chip') and self.current_face.face_chip is not None:
            logger.debug("[FacePanel Prep] Using provided face_chip.") # 修改为 debug
            face_img_to_resize = self.current_face.face_chip.copy()
            if self.current_face.face_box:
                x1, y1, x2, y2 = self.current_face.face_box.to_tlbr()
                ref_crop_width = x2 - x1
                ref_crop_height = y2 - y1
                transform_offset_x = x1
                transform_offset_y = y1
            else:
                ref_crop_width = face_img_to_resize.shape[1]
                ref_crop_height = face_img_to_resize.shape[0]
                logger.warning("[FacePanel Prep] face_box missing, landmark transform might be incorrect when using face_chip.") # 修改为 warning
        
        # --- Fallback: Crop from original frame ---
        elif self.zoom_face and self.current_face.face_box and original_frame is not None:
            logger.debug("[FacePanel Prep] Cropping face from original frame.") # 修改为 debug
            try:
                x1, y1, x2, y2 = map(int, self.current_face.face_box.to_tlbr())
                logger.debug(f"[FacePanel Prep] FaceBox original: ({x1}, {y1}) - ({x2}, {y2})") # 添加日志
                border = int((x2 - x1) * 0.2)
                crop_x1 = max(0, x1 - border)
                crop_y1 = max(0, y1 - border)
                crop_x2 = min(original_frame.shape[1], x2 + border)
                crop_y2 = min(original_frame.shape[0], y2 + border)
                logger.debug(f"[FacePanel Prep] Crop coords calculated: ({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2})") # 添加日志
            
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                     face_img_to_resize = original_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                     logger.debug(f"[FacePanel Prep] Cropped image shape: {face_img_to_resize.shape}") # 添加日志
                     ref_crop_width = crop_x2 - crop_x1 
                     ref_crop_height = crop_y2 - crop_y1
                     transform_offset_x = crop_x1 
                     transform_offset_y = crop_y1
                else:
                    logger.warning("[FacePanel Prep] Invalid crop coordinates, cannot crop.") # 修改为 warning
            except Exception as crop_e:
                logger.error(f"[FacePanel Prep] Error during fallback cropping: {crop_e}")
                face_img_to_resize = None # Ensure it's None on error
        
        # --- Resize the selected image (chip or crop) ---
        if face_img_to_resize is not None and face_img_to_resize.shape[0] > 0 and face_img_to_resize.shape[1] > 0:
            try:
                crop_h, crop_w = face_img_to_resize.shape[:2]
                ratio = min(max_display_width / crop_w, max_display_height / crop_h)
                new_w, new_h = int(crop_w * ratio), int(crop_h * ratio)
                display_width, display_height = new_w, new_h # Store final display size
                self.face_image_display = cv2.resize(face_img_to_resize, (new_w, new_h))
                logger.debug(f"[FacePanel Prep] Resized image shape: {self.face_image_display.shape}") # 添加日志
                # Calculate final scaling factor based on reference crop size and final display size
                transform_scale_x = display_width / ref_crop_width if ref_crop_width > 0 else 1.0
                transform_scale_y = display_height / ref_crop_height if ref_crop_height > 0 else 1.0
            except Exception as resize_e:
                logger.error(f"[FacePanel Prep] Error during resizing: {resize_e}")
                self.face_image_display = None # Ensure it's None on error
                display_width, display_height = 0, 0 # Reset display dims
        else:
            logger.debug("[FacePanel Prep] No valid image to resize.") # 添加日志
            self.face_image_display = None

        # --- Transform landmarks for display ---
        if self.current_face.landmarks and hasattr(self.current_face.landmarks, 'points') and self.current_face.landmarks.points is not None and display_width > 0:
            original_points = self.current_face.landmarks.points
            transformed_points = []
            for point in original_points:
                # Apply offset first, then scale
                x_point = int((point[0] - transform_offset_x) * transform_scale_x)
                y_point = int((point[1] - transform_offset_y) * transform_scale_y)
                transformed_points.append((x_point, y_point))
            self.landmarks_to_draw = np.array(transformed_points)
        else:
             self.landmarks_to_draw = None # Cannot draw landmarks if display failed

        # Always return the calculated (or default) parameters
        # print(f"[FacePanel Prep] Returning: {display_width}, {display_height}, {transform_scale_x:.2f}, {transform_scale_y:.2f}, {transform_offset_x:.2f}, {transform_offset_y:.2f}") # Debug
        return display_width, display_height, transform_scale_x, transform_scale_y, transform_offset_x, transform_offset_y
        
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上 (新布局)
        """
        content_area = self.draw_panel_frame(canvas, x, y, width, height)
        if not self.visible or self.collapsed:
            return
            
        content_x, content_y, content_width, content_height = content_area
        current_y = content_y + 10 # 起始 Y 坐标
        line_height = 18 
        text_offset_x = 10 # 文本左边距 (稍大)
        value_offset_x = text_offset_x + 80 # 值开始的X坐标
        section_spacing = 10 # 各部分之间的垂直间距
        header_font_size = 14
        value_font_size = 12
        
        if not self.face_found or self.current_face is None:
            self.put_text(canvas, "未检测到人脸", 
                         (content_x + text_offset_x, current_y + 15), 
                         self.text_color, 16)
            self.face_image_display = None 
            self.landmarks_to_draw = None
            return
        
        # --- 1. 绘制头部姿态 (Orientation) ---
        self.put_text(canvas, "头部姿态 (Orientation):", (content_x + text_offset_x, current_y), self.highlight_color, header_font_size)
        current_y += line_height + 5
        
        for name, value in self.pose_orientation.items():
            if current_y > content_y + content_height - line_height: break
            label_text = f"{name}:".ljust(10) # 左对齐标签
            value_text = f"{value:.1f}°"
            self.put_text(canvas, label_text, (content_x + text_offset_x, current_y), self.text_color, value_font_size)
            self.put_text(canvas, value_text, (content_x + value_offset_x, current_y), self.text_color, value_font_size)
            current_y += line_height
            
        current_y += section_spacing # 添加间距

        # --- 2. 绘制头部位置 (Pose) ---
        if current_y <= content_y + content_height - (line_height * 4): # 检查空间
            self.put_text(canvas, "头部位置 (Pose):", (content_x + text_offset_x, current_y), self.highlight_color, header_font_size)
            current_y += line_height + 5
            for name, value in self.pose_position.items():
                if current_y > content_y + content_height - line_height: break 
                label_text = f"{name}:".ljust(10)
                value_text = f"{value:.1f}"
                self.put_text(canvas, label_text, (content_x + text_offset_x, current_y), self.text_color, value_font_size)
                self.put_text(canvas, value_text, (content_x + value_offset_x, current_y), self.text_color, value_font_size)
                current_y += line_height
        
        current_y += section_spacing # 添加间距
        
        # --- 3. 绘制置信度 --- 
        if hasattr(self.current_face, 'confidence') and current_y <= content_y + content_height - line_height:
            self.put_text(canvas, "人脸检测置信度:", (content_x + text_offset_x, current_y), self.highlight_color, header_font_size)
            current_y += line_height + 5
            confidence_text = f"{self.current_face.confidence * 100:.1f}%" # 转换为百分比
            self.put_text(canvas, confidence_text, (content_x + text_offset_x, current_y), self.text_color, value_font_size)
            current_y += line_height
            
        final_text_y = current_y + section_spacing # 记录文本块结束位置 (加额外间距)
        
        # --- 4. 准备并绘制人脸图像 (在所有文本下方) ---
        # 调整期望的人脸图像显示大小
        max_face_display_width = content_width - 2 * text_offset_x # 允许左右一点边距
        max_face_display_height = max(50, content_y + content_height - final_text_y - 10) # 使用剩余的垂直空间，至少50像素高
        
        # 使用新的最大尺寸重新准备图像
        display_w, display_h, scale_x, scale_y, offset_x, offset_y = self._prepare_face_image_for_display_v2(self.current_frame, max_face_display_width, max_face_display_height)
        logger.debug(f"[FacePanel Render] Prepared face image: {'Exists' if self.face_image_display is not None else 'None'}, display_dims=({display_w}, {display_h})")

        # --- BEGIN MODIFICATION FOR TypeError ---
        face_image_local = self.face_image_display # Store a local reference
        # --- END MODIFICATION FOR TypeError ---

        if (face_image_local is not None and
           hasattr(face_image_local, 'shape') and
           len(face_image_local.shape) == 3 and
           face_image_local.shape[0] > 0 and
           face_image_local.shape[1] > 0 and
           display_w > 0 and display_h > 0):
            
            # 计算图像绘制位置 (居中或左对齐)
            img_x = content_x + (content_width - display_w) // 2 # 居中对齐 X
            img_y = final_text_y # 紧接文本下方绘制 Y

            draw_h = display_h
            draw_w = display_w
            
            logger.debug(f"[FacePanel Render] Canvas shape: {canvas.shape}, Face img shape: {face_image_local.shape}")
            logger.debug(f"[FacePanel Render] Drawing face at ({img_x}, {img_y}) with target size ({draw_w}, {draw_h})")

            if img_y + draw_h <= canvas.shape[0] and img_x + draw_w <= canvas.shape[1] and img_x >=0 and img_y >=0: # 确保绘制区域在画布内
                try:
                    canvas_slice = canvas[img_y : img_y + draw_h, img_x : img_x + draw_w]
                    # Use the local variable for slicing
                    face_slice_to_draw = face_image_local[0:draw_h, 0:draw_w] 
                    
                    if canvas_slice.shape == face_slice_to_draw.shape:
                        canvas[img_y : img_y + draw_h, img_x : img_x + draw_w] = face_slice_to_draw
                    else:
                        logger.warning(f"[FacePanel Render] Shape mismatch: Canvas slice {canvas_slice.shape}, Face image {face_slice_to_draw.shape}. Attempting resize.")
                        try:
                            # Use the local variable for resizing
                            resized_for_slice = cv2.resize(face_slice_to_draw, (canvas_slice.shape[1], canvas_slice.shape[0]))
                            canvas[img_y : img_y + draw_h, img_x : img_x + draw_w] = resized_for_slice
                        except Exception as e_resize_slice:
                            logger.error(f"[FacePanel Render] Error resizing face image for canvas slice: {e_resize_slice}")
                except Exception as e:
                     logger.error(f"Error drawing face image in FacePanel: {e}", exc_info=True) # 添加 exc_info=True
            else:
                logger.warning(f"[FacePanel Render] Calculated face image position or size is out of canvas bounds or invalid. img_x:{img_x}, img_y:{img_y}, draw_w:{draw_w}, draw_h:{draw_h}, canvas_shape:{canvas.shape}")

            # --- BEGIN MODIFICATION TO HIDE LANDMARKS ---
            # if self.landmarks_to_draw is not None and len(self.landmarks_to_draw) > 0:
            #     for (lx, ly) in self.landmarks_to_draw:
            #         cv2.circle(canvas, (img_x + lx, img_y + ly), 1, (0, 255, 0), -1)
            # --- END MODIFICATION TO HIDE LANDMARKS ---
        elif face_image_local is not None: # Check local variable
             logger.warning(f"[FacePanel Render] face_image_local is present but not a valid image or display_w/h is zero. Type: {type(face_image_local)}, display_w:{display_w}, display_h:{display_h}")
        else:
            logger.debug("[FacePanel Render] face_image_local is None, skipping face image rendering.")

    # --- 需要一个新的 prepare 方法来接受最大尺寸参数 --- 
    def _prepare_face_image_for_display_v2(self, original_frame: Optional[np.ndarray], max_display_width: int, max_display_height: int):
        # 大部分逻辑与原 _prepare_face_image_for_display 类似
        # 主要区别在于使用传入的 max_display_width 和 max_display_height 进行缩放计算
        self.face_image_display = None
        display_width, display_height = 0, 0
        transform_scale_x, transform_scale_y = 1.0, 1.0
        transform_offset_x, transform_offset_y = 0.0, 0.0

        if not self.face_found or self.current_face is None:
            return 0, 0, 1.0, 1.0, 0.0, 0.0

        if original_frame is None:
            logger.debug("[FacePanel Prep V2] original_frame is None.")
            return 0, 0, 1.0, 1.0, 0.0, 0.0

        face_img_to_resize = None
        ref_crop_width, ref_crop_height = 0, 0

        # --- 优先使用 face_chip --- (如果未来实现了对齐)
        if hasattr(self.current_face, 'face_chip') and self.current_face.face_chip is not None:
            logger.debug("[FacePanel Prep V2] Using provided face_chip.")
            face_img_to_resize = self.current_face.face_chip.copy()
            if self.current_face.face_box:
                 x1, y1, x2, y2 = self.current_face.face_box.to_tlbr()
                 ref_crop_width = x2 - x1
                 ref_crop_height = y2 - y1
                 transform_offset_x = x1
                 transform_offset_y = y1
            else:
                 ref_crop_width = face_img_to_resize.shape[1]
                 ref_crop_height = face_img_to_resize.shape[0]
                 logger.warning("[FacePanel Prep V2] face_box missing when using face_chip.")
        
        # --- 回退：从原始帧裁剪 --- 
        elif self.zoom_face and self.current_face.face_box and original_frame is not None:
            logger.debug("[FacePanel Prep V2] Cropping face from original frame.")
            try: 
                x1, y1, x2, y2 = map(int, self.current_face.face_box.to_tlbr())
                border = int((x2 - x1) * 0.2)
                crop_x1 = max(0, x1 - border)
                crop_y1 = max(0, y1 - border)
                crop_x2 = min(original_frame.shape[1], x2 + border)
                crop_y2 = min(original_frame.shape[0], y2 + border)
                if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                     face_img_to_resize = original_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                     ref_crop_width = crop_x2 - crop_x1 
                     ref_crop_height = crop_y2 - crop_y1
                     transform_offset_x = crop_x1 
                     transform_offset_y = crop_y1
                else:
                    logger.warning("[FacePanel Prep V2] Invalid crop coordinates.")
            except Exception as crop_e:
                 logger.error(f"[FacePanel Prep V2] Error during fallback cropping: {crop_e}")
                 face_img_to_resize = None
        
        # --- 调整大小 --- 
        if face_img_to_resize is not None and face_img_to_resize.shape[0] > 0 and face_img_to_resize.shape[1] > 0:
            try: 
                crop_h, crop_w = face_img_to_resize.shape[:2]
                # 使用传入的最大尺寸计算比例
                ratio = min(max_display_width / crop_w, max_display_height / crop_h) if crop_w > 0 and crop_h > 0 else 1.0
                new_w, new_h = int(crop_w * ratio), int(crop_h * ratio)
                # 确保不为0
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                display_width, display_height = new_w, new_h 
                self.face_image_display = cv2.resize(face_img_to_resize, (new_w, new_h))
                logger.debug(f"[FacePanel Prep V2] Resized image shape: {self.face_image_display.shape}")
                transform_scale_x = display_width / ref_crop_width if ref_crop_width > 0 else 1.0
                transform_scale_y = display_height / ref_crop_height if ref_crop_height > 0 else 1.0
            except Exception as resize_e:
                 logger.error(f"[FacePanel Prep V2] Error during resizing: {resize_e}")
                 self.face_image_display = None
                 display_width, display_height = 0, 0
        else:
            logger.debug("[FacePanel Prep V2] No valid image to resize.")
            self.face_image_display = None

        # --- 转换关键点坐标 --- 
        if self.current_face.landmarks and hasattr(self.current_face.landmarks, 'points') and self.current_face.landmarks.points is not None and display_width > 0:
            original_points = self.current_face.landmarks.points
            transformed_points = []
            for point in original_points:
                x_point = int((point[0] - transform_offset_x) * transform_scale_x)
                y_point = int((point[1] - transform_offset_y) * transform_scale_y)
                transformed_points.append((x_point, y_point))
            self.landmarks_to_draw = np.array(transformed_points)
        else:
             self.landmarks_to_draw = None 

        return display_width, display_height, transform_scale_x, transform_scale_y, transform_offset_x, transform_offset_y

    def stop(self):
        """停止面板并取消订阅事件"""
        if self.event_bus:
           self.event_bus.unsubscribe(EventType.FACE_DETECTED, self._on_face_detected)
        super().stop() # 调用父类的 stop 方法

    def update(self, frame: Optional[np.ndarray], **kwargs):
        """更新面板状态，主要是接收最新的视频帧"""
        # 直接更新当前帧的引用
        self.current_frame = frame # 使用传入的帧，可能是 None

    # def toggle_landmarks(self):
    #     ...
    # def toggle_face_box(self):
    #     ...
    # def toggle_face_mesh(self):
    #     ...
    # def toggle_zoom_face(self):
    #     ... 