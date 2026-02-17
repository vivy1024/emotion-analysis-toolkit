#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
姿态估计器
基于 dlib 特征点和 OpenCV solvePnP 实现头部姿态估计
"""

import os
import cv2
import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any

# 导入核心数据类型
from hidden_emotion_detection.core.data_types import FaceDetection, FaceBox, Landmarks, FacePose
from hidden_emotion_detection.core.event_bus import EventBus
# 导入配置管理器 (用于可能的相机参数等)
from hidden_emotion_detection.config import config_manager

# 配置日志 - 移除 basicConfig，依赖 main.py 的配置
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoseEstimator") # 获取名为 "PoseEstimator" 的 logger

class PoseEstimator:
    """姿态估计器，基于 dlib 特征点和 solvePnP"""
    
    def __init__(self):
        """
        初始化姿态估计器
        """
        # 定义标准的 68 点 3D 人脸模型坐标
        # (这些坐标基于通用模型，可能需要微调以获得最佳精度)
        # 参考: https://github.com/dat Tran/face-alignment-dlib/blob/master/shape_predictor_68_face_landmarks.dat.txt
        # 和其他 solvePnP 示例
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # 0 - 鼻尖 (作为原点)
            (0.0, -330.0, -65.0),        # 1 - 右眉外端
            (-225.0, -330.0, -130.0),    # 2 - 右眉内端
            (225.0, -330.0, -130.0),     # 3 - 左眉内端
            (-150.0, -330.0, -125.0),    # 4 - 右眼外角
            (150.0, -330.0, -125.0),     # 5 - 左眼外角
            (0.0, -240.0, -100.0),       # 6 - 鼻子底部
            (-400.0, -170.0, -120.0),    # 7 - 右耳垂
            (400.0, -170.0, -120.0),     # 8 - 左耳垂
            (-170.0, 0.0, -150.0),       # 9 - 嘴右角
            (170.0, 0.0, -150.0),        # 10 - 嘴左角
            (-80.0, 80.0, -150.0),       # 11 - 下巴右侧
            (80.0, 80.0, -150.0),        # 12 - 下巴左侧
            (0.0, 170.0, -150.0),        # 13 - 下巴底部
            (-340.0, -150.0, -80.0),     # 14 - 右脸颊
            (340.0, -150.0, -80.0),      # 15 - 左脸颊
            (0.0, -70.0, -60.0),         # 16 - 鼻梁
            # 以下为 dlib 68 点模型中其他点的估计 3D 坐标
            # (这些是近似值，仅用于 solvePnP 计算)
            (-68.25, -131.25, -79.5),    # 17 - Right eye brow top-outer
            (-135.0, -175.0, -78.75),    # 18 - Right eye brow mid-outer
            (-198.75, -195.0, -87.0),    # 19 - Right eye brow mid-inner
            (-258.75, -183.75, -99.0),   # 20 - Right eye brow top-inner
            (-300.0, -138.75, -105.0),   # 21 - Right eye brow bottom-inner
            (68.25, -131.25, -79.5),     # 22 - Left eye brow top-outer
            (135.0, -175.0, -78.75),     # 23 - Left eye brow mid-outer
            (198.75, -195.0, -87.0),     # 24 - Left eye brow mid-inner
            (258.75, -183.75, -99.0),    # 25 - Left eye brow top-inner
            (300.0, -138.75, -105.0),    # 26 - Left eye brow bottom-inner
            (-30.0, -45.0, -75.0),       # 27 - Nose bridge top
            (-30.0, -15.0, -75.0),       # 28 - Nose bridge mid
            (-30.0, 15.0, -75.0),        # 29 - Nose bridge bottom
            (-30.0, 45.0, -75.0),        # 30 - Nose tip
            (-75.0, -15.0, -78.75),      # 31 - Right nostril wing outer
            (-45.0, -15.0, -78.75),      # 32 - Right nostril wing inner
            (-30.0, -15.0, -78.75),      # 33 - Nose bottom center
            (45.0, -15.0, -78.75),       # 34 - Left nostril wing inner
            (75.0, -15.0, -78.75),       # 35 - Left nostril wing outer
            (-180.0, -105.0, -82.5),     # 36 - Right eye outer corner
            (-142.5, -120.0, -82.5),     # 37 - Right eye top outer
            (-105.0, -123.75, -82.5),    # 38 - Right eye top inner
            (-75.0, -120.0, -82.5),      # 39 - Right eye inner corner
            (-105.0, -105.0, -82.5),     # 40 - Right eye bottom inner
            (-142.5, -101.25, -82.5),    # 41 - Right eye bottom outer
            (75.0, -120.0, -82.5),       # 42 - Left eye inner corner
            (105.0, -123.75, -82.5),     # 43 - Left eye top inner
            (142.5, -120.0, -82.5),      # 44 - Left eye top outer
            (180.0, -105.0, -82.5),      # 45 - Left eye outer corner
            (142.5, -101.25, -82.5),     # 46 - Left eye bottom outer
            (105.0, -105.0, -82.5),      # 47 - Left eye bottom inner
            (-90.0, 30.0, -90.0),        # 48 - Mouth right corner
            (-60.0, 15.0, -90.0),        # 49 - Upper lip top right
            (-30.0, 7.5, -90.0),         # 50 - Upper lip top mid-right
            (0.0, 0.0, -90.0),           # 51 - Upper lip top mid
            (30.0, 7.5, -90.0),          # 52 - Upper lip top mid-left
            (60.0, 15.0, -90.0),         # 53 - Upper lip top left
            (90.0, 30.0, -90.0),         # 54 - Mouth left corner
            (60.0, 45.0, -90.0),         # 55 - Lower lip bottom left
            (30.0, 52.5, -90.0),         # 56 - Lower lip bottom mid-left
            (0.0, 60.0, -90.0),          # 57 - Lower lip bottom mid
            (-30.0, 52.5, -90.0),        # 58 - Lower lip bottom mid-right
            (-60.0, 45.0, -90.0),        # 59 - Lower lip bottom right
            (-60.0, 30.0, -90.0),        # 60 - Upper lip bottom right
            (-30.0, 22.5, -90.0),        # 61 - Upper lip bottom mid-right
            (0.0, 15.0, -90.0),          # 62 - Upper lip bottom mid
            (30.0, 22.5, -90.0),         # 63 - Upper lip bottom mid-left
            (60.0, 30.0, -90.0),         # 64 - Upper lip bottom left
            (-30.0, 60.0, -90.0),        # 65 - Lower lip top right
            (0.0, 67.5, -90.0),          # 66 - Lower lip top mid
            (30.0, 60.0, -90.0)          # 67 - Lower lip top left
        ], dtype=np.float32)

        # 假设相机参数 (可能需要从配置或校准中获取更精确的值)
        # self.focal_length = config_manager.get(...) 
        # self.optical_center = config_manager.get(...)
        # 暂时硬编码
        self.dist_coeffs = np.zeros((4, 1))  # 假设无径向畸变

        logger.info("基于dlib的姿态估计器初始化完成")
    
    def estimate_pose(self, image: np.ndarray, landmarks_2d: np.ndarray) -> Optional[Dict[str, float]]:
        """
        使用solvePnP估计头部姿态
        
        Args:
            image: 输入图像 (用于获取尺寸以估计相机参数)
            landmarks_2d: 检测到的2D面部特征点 (形状为 (N, 2))
            
        Returns:
            Dict[str, float]: 包含欧拉角 (pitch, yaw, roll) 的字典，或在失败时返回 None
        """
        # --- 添加函数入口日志 ---
        logger.debug("[PoseEstimator.estimate_pose] Function entered.")
        # --- 添加函数入口日志 ---
        
        if landmarks_2d is None or len(landmarks_2d) < 68:
            logger.warning(f"[PoseEstimator.estimate_pose] 特征点数量不足 ({len(landmarks_2d) if landmarks_2d is not None else 'None'}), 返回 None.")
            return None
        
        try:
            # 获取图像尺寸
            size = image.shape
            image_h, image_w = size[0], size[1]

            # 估计相机内参矩阵
            focal_length = image_w  # 近似值
            center = (image_w / 2, image_h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            
            # 确保输入点是正确的 numpy 类型
            if not isinstance(landmarks_2d, np.ndarray):
                landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
            else:
                landmarks_2d = landmarks_2d.astype(np.float32)
                
            # 选取用于 solvePnP 的点 (可以使用所有点，或选取的子集)
            # 这里使用 dlib 68 点模型对应的所有点
            points_3d = self.model_points_3d
            points_2d = landmarks_2d

            # --- 添加 solvePnP 调用前日志 ---
            logger.debug("[PoseEstimator.estimate_pose] Preparing to call cv2.solvePnP.")
            # --- 添加 solvePnP 调用前日志 ---

            # 使用 solvePnP 计算旋转和平移向量
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE # 或者 cv2.SOLVEPNP_EPNP
            )

            # --- 添加 solvePnP 调用后日志 (移动原有日志位置) ---
            if not success:
                logger.warning("[PoseEstimator.estimate_pose] cv2.solvePnP 未能成功求解姿态")
                return None
            # --- 添加 solvePnP 调用后日志 ---
                
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # 计算欧拉角
            # 参考: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
            sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6

            if not singular:
                # YXZ 顺序: yaw, pitch, roll
                # 注意：OpenCV 的坐标系和常规航空坐标系可能不同，角度符号和顺序需确认
                # 此处尝试匹配常见的输出习惯 (roll绕x, pitch绕y, yaw绕z - 但从旋转矩阵到欧拉角有多种约定)
                # 假设相机坐标系：X向右，Y向下，Z向前
                
                # Pitch (绕X轴旋转 - 上下点头)
                pitch_rad = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                # Yaw (绕Y轴旋转 - 左右摇头)
                yaw_rad = math.atan2(-rotation_matrix[2, 0], sy)
                # Roll (绕Z轴旋转 - 头部倾斜)
                roll_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                
            else:
                # 奇异情况 (万向锁)
                pitch_rad = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw_rad = math.atan2(-rotation_matrix[2, 0], sy)
                roll_rad = 0
                
                # 将弧度转换为角度
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)
            roll_deg = math.degrees(roll_rad)
                
            # Extract translation vector components
            t_vec = translation_vector.flatten()
            x_pos = t_vec[0]
            y_pos = t_vec[1]
            z_pos = t_vec[2]
            
            # --- 添加日志 ---
            logger.debug(f"[PoseEstimator] solvePnP Success: {success}, Pitch: {pitch_deg:.2f}, Yaw: {yaw_deg:.2f}, Roll: {roll_deg:.2f}, X: {x_pos:.2f}, Y: {y_pos:.2f}, Z: {z_pos:.2f}")
            # --- 添加日志 ---
                
            return {
                "pitch": pitch_deg,
                "yaw": yaw_deg,
                "roll": roll_deg,
                "x_pos": x_pos,
                "y_pos": y_pos,
                "z_pos": z_pos,
                "rvec": rotation_vector,
                "tvec": translation_vector
            }
                
        except Exception as e:
            logger.error(f"使用solvePnP估计姿态失败: {e}")
            import traceback
            traceback.print_exc()
            # --- 添加失败日志 ---
            logger.debug(f"[PoseEstimator] solvePnP failed due to exception.")
            # --- 添加失败日志 ---
            return None
    
    def _analyze_head_pose(self, pitch, yaw, roll):
        """
        分析头部姿态，生成描述性状态
        
        Args:
            pitch: 俯仰角（度）
            yaw: 偏航角（度）
            roll: 翻滚角（度）
            
        Returns:
            Dict[str, Any]: 头部姿态状态描述
        """
        # 姿态阈值（度）
        pitch_threshold = 15.0
        yaw_threshold = 15.0
        roll_threshold = 15.0
        
        # 分析俯仰角（点头）
        if pitch < -pitch_threshold:
            pitch_state = "向下"
        elif pitch > pitch_threshold:
            pitch_state = "向上"
        else:
            pitch_state = "正常"
        
        # 分析偏航角（摇头）
        if yaw < -yaw_threshold:
            yaw_state = "向右"
        elif yaw > yaw_threshold:
            yaw_state = "向左"
        else:
            yaw_state = "正常"
        
        # 分析翻滚角（倾头）
        if roll < -roll_threshold:
            roll_state = "右倾"
        elif roll > roll_threshold:
            roll_state = "左倾"
        else:
            roll_state = "正常"
        
        # 综合分析
        if pitch_state == "正常" and yaw_state == "正常" and roll_state == "正常":
            overall_state = "正视前方"
        else:
            states = []
            if pitch_state != "正常":
                states.append(f"头部{pitch_state}")
            if yaw_state != "正常":
                states.append(f"头部{yaw_state}")
            if roll_state != "正常":
                states.append(f"头部{roll_state}")
            overall_state = "、".join(states)
        
        return {
            "pitch_state": pitch_state,
            "yaw_state": yaw_state,
            "roll_state": roll_state,
            "overall_state": overall_state
        }
    def draw_pose(self, image, pose: FacePose, face_box: FaceBox):
        """
        在图像上绘制姿态估计结果 (使用 FacePose 对象)
        
        Args:
            image: 输入图像
            pose: FacePose 对象，包含 pitch, yaw, roll
            face_box: 人脸边界框 FaceBox 对象
            
        Returns:
            np.ndarray: 绘制了姿态估计结果的图像
        """
        if image is None or pose is None or face_box is None:
            return image
        
        try:
            result = image.copy()
            x, y, w, h = face_box.to_opencv_rect()
            
            # 绘制人脸边界框
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            pitch = pose.pitch
            yaw = pose.yaw
            roll = pose.roll
            
            # 获取头部姿态状态描述
            head_pose_state = self._analyze_head_pose(pitch, yaw, roll)
            overall_state = head_pose_state.get("overall_state", "未知")
            
            # 显示头部姿态信息
            cv2.putText(result, f"头部姿态: {overall_state}", 
                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 显示欧拉角信息
            y_offset = y + h + 15
            cv2.putText(result, f"俯仰角: {pitch:.1f}°", 
                      (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"偏航角: {yaw:.1f}°", 
                      (x, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"翻滚角: {roll:.1f}°", 
                      (x, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # --- 可选：绘制 3D 坐标轴 ---
            # 将旋转向量和平移向量转换回 solvePnP 的输出格式（如果需要精确绘制轴）
            # 或者，直接使用欧拉角近似绘制方向箭头（如下）

            # --- 绘制姿态方向箭头 (基于欧拉角近似) ---
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            arrow_length = 30
            
            # 注意：这里的计算是简化的，仅用于大致指示方向
            # 偏航角（左右） - 影响 X 轴
            dx = math.sin(math.radians(yaw)) * arrow_length * 1.5 # 放大yaw的影响
            # 俯仰角（上下） - 影响 Y 轴
            dy = -math.sin(math.radians(pitch)) * arrow_length * 1.5 # 放大pitch的影响
            
            # 绘制箭头 (从中心指向估计方向)
            # 蓝色 Z 轴 (近似指向观察者)
            cv2.line(result, (face_center_x, face_center_y), 
                     (face_center_x + int(dx), face_center_y + int(dy)), (255, 0, 0), 2)

            # --- 更精确的 3D 轴绘制 (可选) ---
            # 如果需要精确绘制 3D 轴，需要重新进行 solvePnP 或保存其 rvec/tvec
            # ( K = camera_matrix, D = dist_coeffs )
            # axis_points = np.float32([[0,0,0], [100,0,0], [0,100,0], [0,0,100]]) # 轴长100
            # image_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, D)
            # origin = tuple(image_axis_points[0].ravel().astype(int))
            # x_end = tuple(image_axis_points[1].ravel().astype(int))
            # y_end = tuple(image_axis_points[2].ravel().astype(int))
            # z_end = tuple(image_axis_points[3].ravel().astype(int))
            # cv2.line(result, origin, x_end, (0, 0, 255), 3) # X轴 (红)
            # cv2.line(result, origin, y_end, (0, 255, 0), 3) # Y轴 (绿)
            # cv2.line(result, origin, z_end, (255, 0, 0), 3) # Z轴 (蓝)
            
            return result
            
        except Exception as e:
            logger.error(f"绘制姿态估计结果失败: {e}")
            return image


class PoseEstimationEngine:
    """姿态估计引擎，包装PoseEstimator提供引擎接口"""
    
    _instance = None  # 单例实例
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(PoseEstimationEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化姿态估计引擎"""
        if getattr(self, '_initialized', False):
            return
            
        logger.info("初始化姿态估计引擎...")
        
        # 不再需要读取 OpenFace 路径
        # self.config_manager = config_manager
        # openface_path = self.config_manager.get("openface.path") ...
        
        # 创建基于 dlib 的姿态估计器实例
        try:
            self.estimator = PoseEstimator()
        except Exception as e:
            logger.error(f"无法初始化 PoseEstimator: {e}")
            self.estimator = None # 标记为不可用
        
        # 状态变量
        self.is_running = False
        self.is_paused = False
        
        # 获取事件总线实例
        self.event_bus = EventBus()
        
        self._initialized = True
        logger.info("姿态估计引擎初始化完成")
    def start(self):
        """启动姿态估计引擎"""
        if self.is_running:
            logger.warning("姿态估计引擎已经在运行中")
            return
        
        self.is_running = True
        self.is_paused = False
        logger.info("姿态估计引擎已启动")
    
    def stop(self):
        """停止姿态估计引擎"""
        if not self.is_running:
            logger.warning("姿态估计引擎未运行")
            return
        
        self.is_running = False
        logger.info("姿态估计引擎已停止")
    
    def pause(self):
        """暂停姿态估计引擎"""
        if not self.is_running:
            logger.warning("姿态估计引擎未运行")
            return
        
        self.is_paused = True
        logger.info("姿态估计引擎已暂停")
    
    def resume(self):
        """恢复姿态估计引擎"""
        if not self.is_running:
            logger.warning("姿态估计引擎未运行")
            return
        
        self.is_paused = False
        logger.info("姿态估计引擎已恢复")

    
    def estimate_pose(self, image, face_detection: FaceDetection) -> Optional[FacePose]:
        # logger.critical("!!!!!!!!!!!!!!!! PoseEstimationEngine.estimate_pose ENTERED !!!!!!!!!!!!!!!!") # 移除 CRITICAL 日志

        # --- 保持简化逻辑，但移除 print 语句 ---
        pose_angles = None # Default to None
        try:
            if self.estimator and face_detection and face_detection.landmarks:
                landmarks_2d = face_detection.landmarks.points
                if landmarks_2d is not None and len(landmarks_2d) == 68:
                    if not isinstance(landmarks_2d, np.ndarray):
                        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
                    else:
                        landmarks_2d = landmarks_2d.astype(np.float32)
                    pose_angles = self.estimator.estimate_pose(image, landmarks_2d) # Call the underlying estimator

        except Exception as e:
            logger.error(f"调用 self.estimator.estimate_pose 时出错: {e}", exc_info=True) # 保留错误日志
            pose_angles = None # Ensure None on error

        # --- 基于调用结果返回 ---
        if pose_angles:
            pose = FacePose(
                pitch=float(pose_angles.get("pitch", 0.0)),
                yaw=float(pose_angles.get("yaw", 0.0)),
                roll=float(pose_angles.get("roll", 0.0)),
                x_pos=float(pose_angles.get("x_pos", 0.0)),
                y_pos=float(pose_angles.get("y_pos", 0.0)),
                z_pos=float(pose_angles.get("z_pos", 0.0)),
                rvec=pose_angles.get("rvec"),
                tvec=pose_angles.get("tvec")
            )
            return pose
        else:
            return None
    
    def draw_pose(self, image, face_detection: FaceDetection) -> np.ndarray:
        """
        在图像上绘制姿态估计结果 (使用 dlib 方法)
        
        Args:
            image: 输入图像
            face_detection: 人脸检测结果 (包含 pose 和 face_box)
            
        Returns:
            np.ndarray: 绘制了姿态估计结果的图像
        """
        if not self.is_running or image is None or face_detection is None:
            return image
        
        if self.estimator is None:
            return image
        
        # 获取姿态和人脸框
        pose = face_detection.pose
        face_box = face_detection.face_box
        
        if pose is None or face_box is None:
            # 如果没有姿态或人脸框信息，尝试只画框（如果可用）
            if face_box:
                 x, y, w, h = face_box.to_opencv_rect()
                 cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return image
        
        # 调用 PoseEstimator 的 draw_pose 方法
        return self.estimator.draw_pose(image, pose, face_box) 