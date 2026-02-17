import cv2
import numpy as np
import threading
import time
import queue
from typing import Dict, List, Optional, Union, Callable

from .base_detector import BaseAUDetector
from .visualizer import AUVisualizer
from .analyzer import AUAnalyzer

class RealTimeAUProcessor:
    """
    实时AU处理器，用于视频流处理
    """
    
    def __init__(self, 
                 detector: BaseAUDetector,
                 buffer_size: int = 3,
                 skip_frames: int = 1,
                 analysis_window: int = 30):
        """
        初始化实时处理器
        
        参数:
            detector: AU检测器实例
            buffer_size: 处理队列的大小
            skip_frames: 处理时跳过的帧数(1表示每帧都处理)
            analysis_window: 分析窗口大小(帧数)
        """
        self.detector = detector
        self.buffer_size = buffer_size
        self.skip_frames = skip_frames
        
        # 分析器
        self.analyzer = AUAnalyzer(window_size=analysis_window)
        
        # 处理队列和结果 - 增加结果队列大小
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=10)  # 增加结果队列大小从3到10
        
        # 处理线程
        self.processing_thread = None
        self.is_running = False
        self.frame_count = 0
        
        # 回调函数
        self.on_result_callback = None
        
    def start(self, on_result: Optional[Callable] = None):
        """
        启动处理线程
        
        参数:
            on_result: 结果回调函数，签名为 on_result(result: Dict)
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.on_result_callback = on_result
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """停止处理线程"""
        self.is_running = False
        
        if self.processing_thread:
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
                
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        向处理队列添加一帧
        
        参数:
            frame: BGR格式图像
            
        返回:
            是否成功加入队列
        """
        if not self.is_running:
            return False
            
        self.frame_count += 1
        
        # 跳帧处理
        if (self.frame_count - 1) % self.skip_frames != 0:
            return True
            
        # 尝试将帧添加到队列
        try:
            self.frame_queue.put_nowait((self.frame_count, frame.copy()))
            return True
        except queue.Full:
            return False
    
    def get_latest_result(self) -> Optional[Dict]:
        """
        获取最新的处理结果
        
        返回:
            最新结果字典，如果没有则返回None
        """
        if self.result_queue.empty():
            return None
            
        # 清空队列，只保留最新结果
        latest_result = None
        while not self.result_queue.empty():
            try:
                latest_result = self.result_queue.get_nowait()
            except queue.Empty:
                break
                
        return latest_result
    
    def _processing_loop(self):
        """处理线程的主循环"""
        import logging
        logger = logging.getLogger("RealTimeAUProcessor")
        logger.info("处理线程已启动")
        
        while self.is_running:
            try:
                # 从队列获取一帧
                logger.info(f"尝试从队列获取帧，当前队列大小: {self.frame_queue.qsize()}")
                frame_count, frame = self.frame_queue.get(timeout=0.1)
                logger.info(f"从队列获取到帧 #{frame_count}，shape: {frame.shape}")
                
                # 处理帧
                start_time = time.time()
                logger.info(f"开始处理帧 #{frame_count}，调用detector.detect_from_image")
                detection_result = self.detector.detect_from_image(frame)
                logger.info(f"帧 #{frame_count} 处理完成，结果: {detection_result is not None}")
                
                # 如果检测到了人脸
                if detection_result and 'aus' in detection_result:
                    logger.info(f"帧 #{frame_count} 检测到AU，AUs: {list(detection_result['aus'].keys())}")
                    # 更新分析器
                    self.analyzer.update(detection_result['aus'])
                    
                    # 添加处理时间
                    processing_time = time.time() - start_time
                    detection_result['processing_time'] = processing_time
                    
                    # 添加帧信息
                    result = {
                        'frame_count': frame_count,
                        'timestamp': time.time(),
                        **detection_result
                    }
                    
                    # 将结果放入结果队列
                    try:
                        # 如果队列满了，先移除一个老结果再添加新结果
                        if self.result_queue.full():
                            try:
                                self.result_queue.get_nowait()  # 移除队列前面的老结果
                            except queue.Empty:
                                pass
                        # 添加新结果
                        self.result_queue.put_nowait(result)
                        logger.info(f"结果已放入结果队列，当前结果队列大小: {self.result_queue.qsize()}")
                        
                        # 调用回调函数
                        if self.on_result_callback:
                            logger.info(f"调用结果回调函数")
                            self.on_result_callback(result)
                        else:
                            logger.warning(f"没有设置结果回调函数")
                            
                    except queue.Full:
                        logger.warning(f"结果队列已满，尝试强制添加最新结果")
                        # 即使队列满了，也要确保能处理最新结果
                        if self.on_result_callback:
                            self.on_result_callback(result)
                else:
                    logger.warning(f"帧 #{frame_count} 未检测到AU或检测结果为空")
                        
                # 完成此帧处理
                self.frame_queue.task_done()
                logger.info(f"帧 #{frame_count} 处理完成")
                
            except queue.Empty:
                # 队列为空，等待新帧
                logger.debug("队列为空，等待新帧")
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"处理线程错误: {str(e)}", exc_info=True)
                
    def create_visualization(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        创建可视化结果
        
        参数:
            frame: 原始图像
            result: 检测结果
            
        返回:
            可视化后的图像
        """
        vis_img = frame.copy()
        
        if not result or 'aus' not in result:
            return vis_img
            
        # 获取必要的信息
        aus = result.get('aus', {})
        face_box = result.get('face_box')
        emotions = result.get('emotions', {})
        landmarks = result.get('landmarks')
        
        # 绘制AU值
        vis_img = AUVisualizer.visualize_aus(vis_img, aus, face_box)
        
        # 绘制情绪
        if emotions:
            vis_img = AUVisualizer.overlay_emotion_text(vis_img, emotions, (10, 30))
            
        # 绘制特征点
        if landmarks is not None:
            vis_img = AUVisualizer.draw_landmarks(vis_img, landmarks)
            
        # 添加处理时间信息
        processing_time = result.get('processing_time', 0)
        cv2.putText(vis_img, f"Process: {processing_time*1000:.1f}ms", 
                   (10, vis_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
                   
        return vis_img