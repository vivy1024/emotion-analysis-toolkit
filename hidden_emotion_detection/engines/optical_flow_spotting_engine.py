#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
光流微表情 Spotting 引擎
参考 MEGC2024 CCS 冠军方案（中科大），基于多区域光流特征的微表情时间段检测。
纯 CPU 运算，不需要 GPU。

核心思路：
1. 缓冲连续帧和 dlib 68 点关键点
2. 帧间对齐（仿射变换到第一帧坐标系）
3. 计算 7 个 ROI 区域的 Farneback 密集光流
4. 全局运动补偿（减去鼻子中心的全局光流）
5. 基于光流幅度的峰谷检测 + 边界校准
6. 输出微表情时间段 (onset, offset)
"""

import logging
import time
import threading
import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Tuple, Optional

from hidden_emotion_detection.core.base import EventType, MicroEmotionResult
from hidden_emotion_detection.core.event_bus import EventBus

logger = logging.getLogger("OpticalFlowSpottingEngine")


class OpticalFlowSpottingEngine:
    """基于光流的微表情 Spotting 引擎"""

    # ROI 区域定义（基于 dlib 68 点关键点索引）
    ROI_DEFINITIONS = {
        'left_eyebrow': {'landmarks': list(range(17, 22)), 'percent': 0.3},
        'right_eyebrow': {'landmarks': list(range(22, 27)), 'percent': 0.3},
        'left_eye': {'landmarks': list(range(36, 42)), 'percent': 0.2},
        'right_eye': {'landmarks': list(range(42, 48)), 'percent': 0.2},
        'nose': {'landmarks': list(range(27, 36)), 'percent': 0.2},
        'mouth_outer': {'landmarks': list(range(48, 60)), 'percent': 0.3},
        'mouth_inner': {'landmarks': list(range(60, 68)), 'percent': 0.3},
    }

    # 全局运动补偿参考点（鼻子中心）
    GLOBAL_REF_LANDMARKS = [29, 30, 31]

    def __init__(self, event_bus: EventBus, config: dict = None):
        """
        初始化光流 Spotting 引擎

        Args:
            event_bus: 事件总线
            config: 配置字典
        """
        self.event_bus = event_bus
        self.config = config or {}

        # 缓冲区配置
        self.buffer_size = self.config.get('buffer_size', 90)  # 约3秒@30fps
        self.min_frames = self.config.get('min_frames', 30)    # 最少帧数才触发分析
        self.analyze_interval = self.config.get('analyze_interval', 15)  # 每15帧分析一次

        # 光流参数
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        # Spotting 阈值参数（参考 CCS 冠军方案）
        self.peak_threshold = self.config.get('peak_threshold', 0.7)
        self.valley_ratio = self.config.get('valley_ratio', 0.33)
        self.min_duration_frames = self.config.get('min_duration_frames', 3)
        self.max_duration_frames = self.config.get('max_duration_frames', 60)
        self.merge_gap = self.config.get('merge_gap', 3)  # 间隔小于3帧则合并
        self.nms_iou_threshold = self.config.get('nms_iou_threshold', 0.3)

        # 帧缓冲区 {face_id: deque of (gray_frame, landmarks_68)}
        self._frame_buffers: Dict[int, deque] = {}
        self._frame_counters: Dict[int, int] = {}
        self._last_update: Dict[int, float] = {}

        # 检测结果缓冲
        self._active_segments: Dict[int, List] = {}

        # 线程安全
        self._lock = threading.Lock()
        self._running = False

        # 订阅人脸检测事件
        self.event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        logger.info("光流 Spotting 引擎初始化完成")

    def start(self):
        """启动引擎"""
        self._running = True
        logger.info("光流 Spotting 引擎已启动")

    def stop(self):
        """停止引擎"""
        self._running = False
        with self._lock:
            self._frame_buffers.clear()
            self._frame_counters.clear()
        logger.info("光流 Spotting 引擎已停止")

    def _on_face_detected(self, event):
        """处理人脸检测事件"""
        if not self._running:
            return

        try:
            data = event.data if hasattr(event, 'data') else event
            face = data.get('face')
            frame = data.get('frame')
            frame_id = data.get('frame_id', 0)

            if face is None or frame is None:
                return

            # 获取关键点（需要 68 点）
            landmarks = self._get_landmarks_68(face)
            if landmarks is None or len(landmarks) < 68:
                return

            face_id = getattr(face, 'face_id', 0)

            # 转灰度
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            with self._lock:
                # 初始化缓冲区
                if face_id not in self._frame_buffers:
                    self._frame_buffers[face_id] = deque(maxlen=self.buffer_size)
                    self._frame_counters[face_id] = 0

                # 添加到缓冲区
                self._frame_buffers[face_id].append((gray, landmarks, frame_id))
                self._frame_counters[face_id] += 1
                self._last_update[face_id] = time.time()

                # 检查是否触发分析
                buf_len = len(self._frame_buffers[face_id])
                counter = self._frame_counters[face_id]

            if buf_len >= self.min_frames and counter % self.analyze_interval == 0:
                # 在后台线程中分析
                threading.Thread(
                    target=self._analyze_buffer,
                    args=(face_id,),
                    daemon=True
                ).start()

        except Exception as e:
            logger.error(f"处理人脸检测事件失败: {e}")

    def _get_landmarks_68(self, face) -> Optional[np.ndarray]:
        """从 FaceDetection 对象提取 68 点关键点"""
        landmarks = getattr(face, 'landmarks', None)
        if landmarks is None:
            return None

        if isinstance(landmarks, np.ndarray):
            if landmarks.shape[0] >= 68:
                return landmarks[:68].astype(np.float32)
        elif isinstance(landmarks, (list, tuple)):
            if len(landmarks) >= 68:
                return np.array(landmarks[:68], dtype=np.float32)

        return None

    def _analyze_buffer(self, face_id: int):
        """分析帧缓冲区，检测微表情时间段"""
        try:
            with self._lock:
                if face_id not in self._frame_buffers:
                    return
                buffer = list(self._frame_buffers[face_id])

            if len(buffer) < self.min_frames:
                return

            grays = [item[0] for item in buffer]
            landmarks_list = [item[1] for item in buffer]
            frame_ids = [item[2] for item in buffer]

            # 1. 帧对齐
            aligned = self._align_frames(grays, landmarks_list)
            if aligned is None:
                return

            # 2. 计算多区域光流
            roi_flows = self._compute_roi_optical_flows(aligned, landmarks_list)
            if roi_flows is None or len(roi_flows) == 0:
                return

            # 3. 融合多区域光流为单一信号
            fused_signal = self._fuse_roi_signals(roi_flows)

            # 4. 峰谷检测 + 边界校准
            segments = self._detect_segments(fused_signal)

            # 5. NMS 去重
            segments = self._nms_segments(segments)

            # 6. 发布检测结果
            for onset_idx, offset_idx, intensity in segments:
                onset_frame_id = frame_ids[onset_idx] if onset_idx < len(frame_ids) else 0
                offset_frame_id = frame_ids[offset_idx] if offset_idx < len(frame_ids) else 0
                duration_frames = offset_idx - onset_idx

                # 按持续时间分类：微表情 < 0.5秒（约15帧@30fps）
                fps = self.config.get('fps', 30)
                duration_sec = duration_frames / fps
                expr_type = "micro" if duration_sec <= 0.5 else "macro"

                result_data = {
                    'face_id': face_id,
                    'onset_frame': onset_frame_id,
                    'offset_frame': offset_frame_id,
                    'duration_frames': duration_frames,
                    'duration_sec': duration_sec,
                    'intensity': float(intensity),
                    'expression_type': expr_type,
                    'source': 'optical_flow_spotting'
                }

                self.event_bus.publish(EventType.MICRO_RESULT, result_data)
                logger.debug(
                    f"检测到{expr_type}表情: face={face_id}, "
                    f"frames={onset_frame_id}-{offset_frame_id}, "
                    f"duration={duration_sec:.2f}s, intensity={intensity:.3f}"
                )

        except Exception as e:
            logger.error(f"分析缓冲区失败 (face_id={face_id}): {e}")

    def _align_frames(self, grays: List[np.ndarray],
                      landmarks_list: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """
        帧序列对齐（仿射变换到第一帧坐标系）
        参考 CCS 冠军 align.py
        """
        if len(grays) < 2:
            return None

        ref_lm = landmarks_list[0]
        # 参考点：左眼中心(39)、右眼中心(42)、鼻尖(33)
        src_pts = np.float32([ref_lm[39], ref_lm[42], ref_lm[33]])

        aligned = [grays[0]]  # 第一帧不需要对齐
        h, w = grays[0].shape[:2]

        for i in range(1, len(grays)):
            lm = landmarks_list[i]
            dst_pts = np.float32([lm[39], lm[42], lm[33]])

            try:
                M = cv2.getAffineTransform(dst_pts, src_pts)
                warped = cv2.warpAffine(
                    grays[i], M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                aligned.append(warped)
            except cv2.error:
                aligned.append(grays[i])  # 对齐失败则用原图

        return aligned

    def _compute_roi_optical_flows(self, aligned: List[np.ndarray],
                                    landmarks_list: List[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """
        计算各 ROI 区域的光流幅度序列
        参考 CCS 冠军的 draw_roiline19 核心逻辑
        """
        n_frames = len(aligned)
        if n_frames < 2:
            return None

        roi_flows = {name: np.zeros(n_frames - 1) for name in self.ROI_DEFINITIONS}
        global_flows = np.zeros((n_frames - 1, 2))  # 全局光流 (x, y)

        for i in range(n_frames - 1):
            # 计算帧间光流
            flow = cv2.calcOpticalFlowFarneback(
                aligned[i], aligned[i + 1], None, **self.flow_params
            )

            # 全局运动（鼻子中心区域）
            ref_lm = landmarks_list[min(i, len(landmarks_list) - 1)]
            gx, gy = self._get_roi_flow(
                flow, ref_lm, self.GLOBAL_REF_LANDMARKS, percent=0.5, expand=10
            )
            global_flows[i] = [gx, gy]

            # 各 ROI 区域光流（减去全局运动）
            for roi_name, roi_def in self.ROI_DEFINITIONS.items():
                lx, ly = self._get_roi_flow(
                    flow, ref_lm, roi_def['landmarks'],
                    percent=roi_def['percent'], expand=5
                )
                # 运动补偿
                comp_x = lx - gx
                comp_y = ly - gy
                magnitude = np.sqrt(comp_x ** 2 + comp_y ** 2)
                roi_flows[roi_name][i] = magnitude

        return roi_flows

    def _get_roi_flow(self, flow: np.ndarray, landmarks: np.ndarray,
                      landmark_indices: List[int], percent: float = 0.3,
                      expand: int = 5) -> Tuple[float, float]:
        """
        提取指定 ROI 区域的平均光流（取高幅度部分）

        Args:
            flow: 光流场 (H, W, 2)
            landmarks: 68 点关键点
            landmark_indices: ROI 对应的关键点索引
            percent: 取光流幅度最大的百分比
            expand: ROI 扩展像素数
        """
        h, w = flow.shape[:2]

        # 获取 ROI 边界
        pts = landmarks[landmark_indices].astype(int)
        x_min = max(0, pts[:, 0].min() - expand)
        x_max = min(w, pts[:, 0].max() + expand)
        y_min = max(0, pts[:, 1].min() - expand)
        y_max = min(h, pts[:, 1].max() + expand)

        if x_max <= x_min or y_max <= y_min:
            return 0.0, 0.0

        # 提取 ROI 区域光流
        roi_flow = flow[y_min:y_max, x_min:x_max]
        fx = roi_flow[:, :, 0].flatten()
        fy = roi_flow[:, :, 1].flatten()
        mag = np.sqrt(fx ** 2 + fy ** 2)

        if len(mag) == 0:
            return 0.0, 0.0

        # 取高幅度部分的平均值
        threshold = np.percentile(mag, (1 - percent) * 100)
        mask = mag >= threshold

        if mask.sum() == 0:
            return float(fx.mean()), float(fy.mean())

        return float(fx[mask].mean()), float(fy[mask].mean())

    def _fuse_roi_signals(self, roi_flows: Dict[str, np.ndarray]) -> np.ndarray:
        """融合多区域光流信号为单一检测信号"""
        signals = list(roi_flows.values())
        if not signals:
            return np.array([])

        # 取所有 ROI 的最大值作为融合信号
        stacked = np.stack(signals, axis=0)
        fused = np.max(stacked, axis=0)

        return fused

    def _detect_segments(self, signal: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        基于光流信号检测微表情时间段
        两阶段：粗检测 + 边界校准

        Returns:
            List of (onset_idx, offset_idx, intensity)
        """
        if len(signal) < self.min_duration_frames:
            return []

        segments = []

        # 第一阶段：粗检测 - 找超过阈值的连续区域
        above_threshold = signal > self.peak_threshold
        in_segment = False
        onset = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_segment:
                onset = i
                in_segment = True
            elif not above_threshold[i] and in_segment:
                # 检查间隔是否小于 merge_gap
                gap_end = min(i + self.merge_gap, len(above_threshold))
                if any(above_threshold[i:gap_end]):
                    continue  # 间隔内还有高值，继续
                offset = i
                intensity = float(signal[onset:offset].max())
                segments.append((onset, offset, intensity))
                in_segment = False

        # 处理末尾
        if in_segment:
            offset = len(signal) - 1
            intensity = float(signal[onset:offset + 1].max())
            segments.append((onset, offset, intensity))

        # 第二阶段：边界校准
        calibrated = []
        for onset, offset, intensity in segments:
            duration = offset - onset
            if duration < self.min_duration_frames:
                continue
            if duration > self.max_duration_frames:
                continue

            # 向前扩展：找到信号开始上升的点
            new_onset = onset
            search_start = max(0, onset - 10)
            if search_start < onset:
                local_min_idx = search_start + np.argmin(signal[search_start:onset])
                if signal[onset] - signal[local_min_idx] > self.valley_ratio * intensity:
                    new_onset = local_min_idx

            # 向后扩展：找到信号回落的点
            new_offset = offset
            search_end = min(len(signal), offset + 10)
            if offset < search_end:
                local_min_idx = offset + np.argmin(signal[offset:search_end])
                if signal[offset - 1] - signal[local_min_idx] > self.valley_ratio * intensity:
                    new_offset = local_min_idx

            calibrated.append((new_onset, new_offset, intensity))

        return calibrated

    def _nms_segments(self, segments: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """非极大值抑制，去除重叠的检测段"""
        if len(segments) <= 1:
            return segments

        # 按强度降序排列
        segments = sorted(segments, key=lambda x: x[2], reverse=True)
        keep = []

        for seg in segments:
            overlap = False
            for kept in keep:
                iou = self._compute_iou(seg, kept)
                if iou > self.nms_iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(seg)

        return keep

    @staticmethod
    def _compute_iou(seg1: Tuple[int, int, float],
                     seg2: Tuple[int, int, float]) -> float:
        """计算两个时间段的 IoU"""
        s1, e1 = seg1[0], seg1[1]
        s2, e2 = seg2[0], seg2[1]

        intersection = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)

        if union <= 0:
            return 0.0
        return intersection / union

    def cleanup_stale_buffers(self, timeout: float = 60.0):
        """清理超时的缓冲区"""
        now = time.time()
        with self._lock:
            stale_ids = [
                fid for fid, t in self._last_update.items()
                if now - t > timeout
            ]
            for fid in stale_ids:
                self._frame_buffers.pop(fid, None)
                self._frame_counters.pop(fid, None)
                self._last_update.pop(fid, None)
                self._active_segments.pop(fid, None)
                logger.debug(f"清理超时缓冲区: face_id={fid}")
