#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 视频显示面板
显示实时视频流和人脸检测结果
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List

from .base_panel import BasePanel
from ..core.data_types import FrameResult, FaceDetection, Event, EventType
from ..core.event_bus import EventBus

logger = logging.getLogger(__name__)

class VideoPanel(BasePanel):
    """视频显示面板，实时视频+人脸68点位"""
    
    def __init__(self):
        """初始化视频面板"""
        super().__init__("实时视频检测")
        self.current_frame: Optional[np.ndarray] = None
        self.faces_to_draw: List[FaceDetection] = [] # Store faces for drawing
        self.fps = 0.0
        # --- Subscribe to face detection events ---
        self.event_bus = EventBus()
        self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        # We also need the frame data, which might come via update or event
        # Let's assume update provides the main frame
        
        # 添加最大跟踪人脸数限制
        self.max_faces_to_draw = 5  # 最多同时显示5个人脸
    def _on_face_detected(self, event: Event):
        """ Handle face detection events to get face data for drawing """
        event_data = event.data
        if isinstance(event_data, dict):
            face = event_data.get('face')
            # Store the latest face data. Since rendering happens later,
            # we might need a mechanism to associate faces with the correct frame
            # For simplicity now, just store the latest detected face(s).
            # A better approach might be to store faces per frame_id if available.
            # Or rely on the FrameResult passed to update.
            # Let's clear and add, assuming update will provide the main list
            # self.faces_to_draw.clear() # Clear old faces on new event? Risky.
            if face:
                # Find if this face ID already exists and update it, otherwise add
                updated = False
                for i, existing_face in enumerate(self.faces_to_draw):
                    if existing_face.face_id == face.face_id:
                        self.faces_to_draw[i] = face
                        updated = True
                        break
                if not updated:
                     # Limit the number of tracked faces to draw if necessary
                     # if len(self.faces_to_draw) < MAX_FACES_TO_DRAW:
                         self.faces_to_draw.append(face)
        else:
            logger.warning("VideoPanel received FACE_DETECTED event with unexpected data type.")

    def update(self, frame: Optional[np.ndarray], result: Optional[FrameResult], fps: float):
        """ Update with the latest frame and analysis results """
        self.current_frame = frame.copy() if frame is not None else None # Store a copy
        self.fps = fps
        # Use faces from FrameResult if available, potentially overwriting event-based ones
        if result and result.face_detections:
            self.faces_to_draw = result.face_detections[:] # Use a copy of the list
        elif frame is None: # If no frame and no result, clear faces
            self.faces_to_draw = []
    def render(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """
        将面板渲染到画布上
        
        Args:
            canvas: 要渲染的画布
            x: 面板左上角x坐标
            y: 面板左上角y坐标
            width: 面板宽度
            height: 面板高度
        """
        content_area = self.draw_panel_frame(canvas, x, y, width, height)
        if not self.visible or self.collapsed:
            return

        content_x, content_y, content_width, content_height = content_area
        
        if self.current_frame is not None:
            # --- Draw the video frame ---
            frame_h, frame_w = self.current_frame.shape[:2]
            target_aspect_ratio = content_width / content_height
            frame_aspect_ratio = frame_w / frame_h

            if frame_aspect_ratio > target_aspect_ratio: # Frame is wider, fit width
                draw_w = content_width
                draw_h = int(draw_w / frame_aspect_ratio)
            else: # Frame is taller or equal, fit height
                draw_h = content_height
                draw_w = int(draw_h * frame_aspect_ratio)

            # Center the frame
            draw_x = content_x + (content_width - draw_w) // 2
            draw_y = content_y + (content_height - draw_h) // 2

            # Resize and draw
            try:
                resized_frame = cv2.resize(self.current_frame, (draw_w, draw_h))

                # --- Draw faces, landmarks, and pose on the resized_frame BEFORE placing it ---
                scale_x = draw_w / frame_w
                scale_y = draw_h / frame_h

                for face in self.faces_to_draw:
                    # Draw face box (optional, maybe disable later)
                    if face.face_box:
                        x1, y1, x2, y2 = face.face_box.to_tlbr()
                        # Scale coordinates to the resized frame
                        sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                        sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                        cv2.rectangle(resized_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1) # Green box

                    # Draw landmarks
                    if face.landmarks and face.landmarks.points is not None:
                        for point in face.landmarks.points:
                            # Scale point coordinates
                            sx, sy = int(point[0] * scale_x), int(point[1] * scale_y)
                            # Draw on resized_frame
                            cv2.circle(resized_frame, (sx, sy), 1, (0, 0, 255), -1) # Red points

                    # Draw pose axis
                    if face.pose and face.pose.rvec is not None and face.pose.tvec is not None and face.landmarks and face.landmarks.points is not None:
                         try:
                            # Use approximate camera matrix based on original frame size
                            cam_matrix = np.array([[frame_w, 0, frame_w/2],
                                                   [0, frame_w, frame_h/2], # Use frame_w also for fy approx
                                                   [0, 0, 1]], dtype = "double")
                            dist_coeffs = np.zeros((4,1))
                            axis = np.float32([[100,0,0], [0,100,0], [0,0,-100]]).reshape(-1,3) # Shorter axis
                            imgpts, jac = cv2.projectPoints(axis, face.pose.rvec, face.pose.tvec, cam_matrix, dist_coeffs)

                            # Get nose tip (point 30) coordinates on original frame
                            nose_tip_orig = face.landmarks.points[30]
                            # Scale nose tip coordinates to resized frame
                            nose_tip_scaled = (int(nose_tip_orig[0] * scale_x), int(nose_tip_orig[1] * scale_y))
        
                            # Scale projected points to resized frame
                            pt0_scaled = tuple( (imgpts[0].ravel() * [scale_x, scale_y]).astype(int) )
                            pt1_scaled = tuple( (imgpts[1].ravel() * [scale_x, scale_y]).astype(int) )
                            pt2_scaled = tuple( (imgpts[2].ravel() * [scale_x, scale_y]).astype(int) )

                            # Draw lines on resized_frame
                            cv2.line(resized_frame, nose_tip_scaled, pt0_scaled, (255,0,0), 2) # X Blue
                            cv2.line(resized_frame, nose_tip_scaled, pt1_scaled, (0,255,0), 2) # Y Green
                            cv2.line(resized_frame, nose_tip_scaled, pt2_scaled, (0,0,255), 2) # Z Red
                         except Exception as pose_draw_e:
                              logger.error(f"Error drawing pose axis: {pose_draw_e}")
            
                # Place the resized frame (with drawings) onto the canvas
                canvas[draw_y : draw_y + draw_h, draw_x : draw_x + draw_w] = resized_frame

            except Exception as e:
                logger.error(f"Error resizing or drawing video frame: {e}")
                # Optionally draw an error message
                self.put_text(canvas, "Frame Error", (content_x + 5, content_y + 20), (0, 0, 255), 14)

            # --- Draw FPS counter ---
            fps_text = f"FPS: {self.fps:.1f}"
            self.put_text(canvas, fps_text, (content_x + 5, content_y + 20), (0, 255, 0), 14)
        else:
            # No frame available
             self.put_text(canvas, "No Video Feed", (content_x + 10, content_y + 30), (150, 150, 150), 16)

    def stop(self):
        """ Stop panel and unsubscribe events """
        self.event_bus.unsubscribe(EventType.FACE_DETECTED, self._on_face_detected)
        super().stop() 