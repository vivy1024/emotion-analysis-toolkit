#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 应用主类

这个模块提供了系统的主应用类，负责初始化和管理所有组件。
"""

import os
import cv2
import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import numpy as np
import sys

# 导入核心组件
from hidden_emotion_detection.core.event_bus import EventBus, EventType, Event
from hidden_emotion_detection.core.data_types import (
    FrameResult, FaceDetection, EmotionResult, AUResult, HiddenEmotionResult
)
from hidden_emotion_detection.core.frame import Frame

# 导入配置管理器和模型配置类
from hidden_emotion_detection.config import config_manager
from hidden_emotion_detection.config.models import ModelsConfig # 导入 ModelsConfig 类
# from enhance_hidden.config.models import models_config # 不再导入实例

# 导入引擎组件
from hidden_emotion_detection.engines.face_detection_engine import FaceDetectionEngine
from hidden_emotion_detection.engines.macro_emotion_engine import MacroEmotionEngine
from hidden_emotion_detection.engines.micro_emotion_engine import MicroEmotionEngine
# from enhance_hidden.engines.emotion_analysis_engine import EmotionAnalysisEngine
# from enhance_hidden.engines.speech_to_text_engine import SpeechToTextEngine
from hidden_emotion_detection.engines.au_engine import AUEngine
from hidden_emotion_detection.engines.hidden_emotion_engine import HiddenEmotionEngine
from hidden_emotion_detection.engines.pose_estimator import PoseEstimationEngine

# 导入UI组件
from hidden_emotion_detection.ui.layout_manager import EnhancedUILayout

# 导入反馈机制
# from enhance_hidden.feedback_mechanism import FeedbackMechanism # 已注释掉


class HiddenEmotionApp:
    """
    隐藏情绪检测应用主类
    
    负责初始化和管理所有组件，包括视频捕获、引擎处理和UI显示。
    """
    
    def __init__(self, 
                 video_source=0, 
                 width=1280, 
                 height=720, 
                 models_dir=None,
                 fps_limit=30,
                 on_emotion_callback: Callable[[Dict], None] = None):
        """
        初始化应用
        
        Args:
            video_source: 视频源，可以是摄像头索引或视频文件路径
            width: 窗口宽度
            height: 窗口高度
            models_dir: 模型文件目录
            fps_limit: 帧率限制
            on_emotion_callback: 情绪检测回调函数
        """
        self.logger = logging.getLogger('enhance_hidden.app')
        self.logger.setLevel(logging.DEBUG) # 设置日志级别为DEBUG
        self.logger.info("初始化隐藏情绪检测应用...")
        
        # 保存参数
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.on_emotion_callback = on_emotion_callback
        
        # 初始化状态变量
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        
        # 获取全局配置实例
        self.config_manager = config_manager
        
        # 使用 config_manager 创建 ModelsConfig 实例
        self.models_config = ModelsConfig(self.config_manager)
        
        # 检查并应用自定义模型目录 (如果需要，但 ModelsConfig 内部会处理)
        self.models_dir = models_dir
        
        # 初始化事件总线
        self.event_bus = EventBus()
        
        # 初始化引擎组件
        self._init_engines()
        
        # 初始化UI组件
        self._init_ui(width, height)
        
        # 初始化视频捕获
        self._init_video_capture()
        
        # 注册事件处理器
        self._register_event_handlers()
        
        self.logger.info("应用初始化完成")
    
    def _init_engines(self):
        """
        初始化引擎组件
        """
        self.logger.info("初始化引擎组件...")
        
        # 初始化人脸检测引擎 (总是需要)
        self.face_engine = FaceDetectionEngine(config=self.config_manager)
        self.logger.debug("[App Init] FaceDetectionEngine created.")
        
        # 重新排序引擎初始化

        # 初始化AU分析引擎
        is_au_enabled = self.config_manager.get("au.enabled", True)
        self.logger.debug(f"[App Init] au.enabled config value: {is_au_enabled}")
        if is_au_enabled:
            try:
                self.au_engine = AUEngine(self.config_manager, self.event_bus)
                self.logger.info("[App Init] AUEngine instance created.")
            except Exception as e:
                self.au_engine = None
                self.logger.error(f"[App Init] Failed to create AUEngine instance: {e}", exc_info=True)
        else:
            self.au_engine = None
            self.logger.info("AU分析引擎已禁用，跳过初始化。")

        # 初始化AU辅助情绪引擎 (在创建宏观和微表情引擎之前)
        # AUEmotionEngine 已被删除，相关逻辑移除
        # is_au_emotion_enabled = self.config_manager.get("au.enabled", True)
        # self.logger.debug(f"[App Init] au_emotion.enabled (using au.enabled) config value: {is_au_emotion_enabled}")
        # if is_au_emotion_enabled:
        #     try:
        #         self.au_emotion_engine = AUEmotionEngine(self.event_bus)
        #         self.logger.info("[App Init] AUEmotionEngine instance created.")
        #     except Exception as e:
        #         self.au_emotion_engine = None
        #         self.logger.error(f"[App Init] Failed to create AUEmotionEngine instance: {e}", exc_info=True)
        # else:
        #     self.au_emotion_engine = None
        #     self.logger.info("AU辅助情绪引擎已禁用，跳过初始化。")
        
        # 初始化宏观情绪分析引擎
        is_macro_enabled = self.config_manager.get("macro.enabled", True)
        self.logger.debug(f"[App Init] macro.enabled config value: {is_macro_enabled}")
        if is_macro_enabled:
            try:
                # 将 au_emotion_engine 替换为 au_engine
                # 注意：MacroEmotionEngine 可能需要调整以适应新的 au_engine 接口
                self.macro_engine = MacroEmotionEngine(self.config_manager, self.event_bus, self.au_engine)
                self.logger.info("[App Init] MacroEmotionEngine instance created.")
            except Exception as e:
                self.macro_engine = None
                self.logger.error(f"[App Init] Failed to create MacroEmotionEngine instance: {e}", exc_info=True)
        else:
            self.macro_engine = None
            self.logger.info("宏观表情引擎已禁用，跳过初始化。")
        
        # 初始化微表情分析引擎
        is_micro_enabled = self.config_manager.get("micro.enabled", True)
        self.logger.debug(f"[App Init] micro.enabled config value: {is_micro_enabled}")
        if is_micro_enabled:
            try:
                # 将 au_emotion_engine 替换为 au_engine
                # 注意：MicroEmotionEngine 可能需要调整以适应新的 au_engine 接口
                self.micro_engine = MicroEmotionEngine(self.config_manager, self.event_bus, self.au_engine)
                self.logger.info("[App Init] MicroEmotionEngine instance created.")
            except Exception as e:
                self.micro_engine = None
                self.logger.error(f"[App Init] Failed to create MicroEmotionEngine instance: {e}", exc_info=True)
        else:
            self.micro_engine = None
            self.logger.info("微表情引擎已禁用，跳过初始化。")
            
        # 初始化隐藏情绪分析引擎
        is_hidden_enabled = self.config_manager.get("hidden.enabled", True)
        self.logger.debug(f"[App Init] hidden.enabled config value: {is_hidden_enabled}")
        if is_hidden_enabled:
            try:
                self.hidden_engine = HiddenEmotionEngine()
                self.logger.info("[App Init] HiddenEmotionEngine instance created.")
            except Exception as e:
                self.hidden_engine = None
                self.logger.error(f"[App Init] Failed to create HiddenEmotionEngine instance: {e}", exc_info=True)
        else:
            self.hidden_engine = None
            self.logger.info("隐藏情绪引擎已禁用，跳过初始化。")
    
    def _init_ui(self, width, height):
        """
        初始化UI组件
        """
        self.logger.info("初始化UI组件...")
        try:
            # 创建UI布局管理器，并传递 event_bus
            self.ui = EnhancedUILayout(self.event_bus, "增强版隐藏情绪检测系统", width, height)
        except ImportError as e:
            # --- BEGIN TEMPORARY DEBUGGING ---            
            import sys
            import traceback
            print(f"DEBUG APP _init_ui: Caught ImportError: {e!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # --- END TEMPORARY DEBUGGING ---
            self.logger.error(f"UI组件导入失败: {e}", exc_info=True) # 保持原有日志
            # 根据错误信息选择性处理，或提供一个更通用的回退UI
            if "AUStatusPanel" in str(e) or "au_status_panel" in str(e).lower(): # Condition D
                # self.logger.warning("AU状态面板导入失败") # Line E - Temporarily commented out
                pass # Explicitly do nothing if condition is met, for clarity during test
            # 这里可以添加一个更简单的回退UI，或者允许程序在没有某些面板的情况下继续运行
            # self.ui = FallbackUILayout(self.event_bus, "增强版隐藏情绪检测系统 - 回退模式", width, height)
            # raise # 或者重新引发异常，如果UI是关键组件
        except Exception as e:
            self.logger.error(f"初始化UI时发生一般错误: {e}", exc_info=True)
            # 这里也可以考虑回退UI或退出
            raise # 重新引发，以便主程序可以捕获
    def _init_video_capture(self):
        """
        初始化视频捕获
        """
        self.logger.info(f"初始化视频捕获，源: {self.video_source}")
        
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开视频源: {self.video_source}")
                raise ValueError(f"无法打开视频源: {self.video_source}")
            
            # 获取视频信息
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"视频尺寸: {self.frame_width}x{self.frame_height}, FPS: {self.video_fps}")
            
        except Exception as e:
            self.logger.error(f"初始化视频捕获时出错: {e}")
            raise
    
    def _register_event_handlers(self):
        """
        注册事件处理器
        """
        # 注册自定义回调
        if self.on_emotion_callback:
            self.event_bus.subscribe(EventType.HIDDEN_EMOTION_ANALYZED, self._on_emotion_callback)
    
    def _on_emotion_callback(self, event: Event):
        """
        情绪检测回调处理
        """
        if self.on_emotion_callback and 'result' in event.data:
            self.on_emotion_callback(event.data['result'])
    
    def start(self):
        """
        启动应用
        """
        if self.running:
            self.logger.warning("应用已经在运行")
            return
        
        self.logger.info("启动应用...")
        
        # 启动事件总线
        self.event_bus.start()
        
        # 启动引擎组件 (检查是否为 None)
        self.logger.debug("[App Start] Starting face_engine...")
        self.face_engine.start()
        
        self.logger.debug(f"[App Start] Checking macro_engine (exists: {self.macro_engine is not None})...")
        if self.macro_engine: 
            self.macro_engine.start()
            self.logger.info("[App Start] macro_engine started.")
            
        self.logger.debug(f"[App Start] Checking micro_engine (exists: {self.micro_engine is not None})...")
        if self.micro_engine: 
            self.micro_engine.start()
            self.logger.info("[App Start] micro_engine started.")
            
        self.logger.debug(f"[App Start] Checking au_engine (exists: {self.au_engine is not None})...")
        if self.au_engine: 
            self.au_engine.start()
            self.logger.info("[App Start] au_engine started.")
            
        self.logger.debug(f"[App Start] Checking hidden_engine (exists: {self.hidden_engine is not None})...")
        if self.hidden_engine: 
            self.hidden_engine.start()
            self.logger.info("[App Start] hidden_engine started.")
        
        # 设置运行标志
        self.running = True
        self.paused = False
        
        # 启动主循环
        self._main_loop()
    
    def stop(self):
        """
        停止应用
        """
        if not self.running:
            return
        
        self.logger.info("停止应用...")
        
        # 设置运行标志
        self.running = False
        
        # 停止引擎组件 (检查是否为 None)
        self.logger.debug("Stopping face_engine...")
        self.face_engine.stop()
        
        self.logger.debug(f"Checking macro_engine for stop (exists: {self.macro_engine is not None})...")
        if self.macro_engine: self.macro_engine.stop()
        
        self.logger.debug(f"Checking micro_engine for stop (exists: {self.micro_engine is not None})...")
        if self.micro_engine: self.micro_engine.stop()
        
        self.logger.debug(f"Checking au_engine for stop (exists: {self.au_engine is not None})...")
        if self.au_engine: self.au_engine.stop()

        # 注意：AUEmotionEngine 通常不需要显式 stop()
        # 如果它有自己的线程或需要清理资源，则需要添加 self.au_emotion_engine.stop() # 已移除

        self.logger.debug(f"Checking hidden_engine for stop (exists: {self.hidden_engine is not None})...")
        if self.hidden_engine: self.hidden_engine.stop()
        
        # 停止事件总线
        self.event_bus.stop()
        
        # 释放视频捕获
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        self.logger.info("应用已停止")
    
    def pause(self):
        """
        暂停应用
        """
        if not self.running or self.paused:
            return
        
        self.logger.info("暂停应用...")
        
        # 设置暂停标志
        self.paused = True
        
        # 暂停引擎组件 (检查是否为 None)
        if self.hidden_engine: self.hidden_engine.pause()
        if self.au_engine: self.au_engine.pause()
        if self.micro_engine: self.micro_engine.pause()
        if self.macro_engine: self.macro_engine.pause()
        self.face_engine.pause()
    
    def resume(self):
        """
        恢复应用
        """
        if not self.running or not self.paused:
            return
        
        self.logger.info("恢复应用...")
        
        # 设置暂停标志
        self.paused = False
        
        # 恢复引擎组件 (检查是否为 None)
        self.face_engine.resume()
        if self.macro_engine: self.macro_engine.resume()
        if self.micro_engine: self.micro_engine.resume()
        if self.au_engine: self.au_engine.resume()
        if self.hidden_engine: self.hidden_engine.resume()
    
    def _main_loop(self):
        """
        主循环
        """
        self.logger.info("进入主循环...")
        
        while self.running:
            try:
                # 计算帧率控制
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # 帧率限制
                if self.fps_limit > 0 and elapsed < 1.0 / self.fps_limit:
                    delay = max(1, int((1.0 / self.fps_limit - elapsed) * 1000))
                    key = cv2.waitKey(delay)
                    if key == 27:  # ESC键
                        break
                    elif key == ord(' '):  # 空格键
                        if self.paused:
                            self.resume()
                        else:
                            self.pause()
                    continue
                
                # 更新帧率计算
                if elapsed > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / elapsed) if self.fps > 0 else 1.0 / elapsed
                self.last_frame_time = current_time
                
                # 如果暂停，只处理UI更新和按键
                if self.paused:
                    self.ui.render()
                    self.ui.show()
                    key = cv2.waitKey(100)
                    if key == 27:  # ESC键
                        break
                    elif key == ord(' '):  # 空格键
                        self.resume()
                    continue
                
                # 读取视频帧
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning(f"主循环：cap.read() 返回 False，视频源结束或错误。")
                    break
                else:
                    # 可以选择性地在这里加一个 debug 日志确认帧读取成功
                    self.logger.debug(f"主循环：成功读取帧，Shape: {frame.shape}")
                
                # 处理视频帧
                self._process_frame(frame)
                
                # 处理按键
                key = cv2.waitKey(1)
                if key == 27:  # ESC键
                    break
                elif key == ord(' '):  # 空格键
                    self.pause()
                
            except Exception as e:
                self.logger.error(f"主循环中出错: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 停止应用
        self.stop()
    
    def _process_frame(self, frame):
        """
        处理视频帧
        """
        # 增加帧计数
        self.frame_count += 1

        # 创建帧结果对象
        frame_result = FrameResult(frame_id=self.frame_count)

        # 检测人脸
        faces = self.face_engine.detect_faces(frame)
        frame_result.face_detections = faces # 直接将引擎返回的列表赋值给帧结果

        # 遍历检测到的有效人脸，发布事件
        # 注意：现在 face_engine 返回的 faces 列表已经包含了 pose (如果启用并成功)
        for face in faces: # face 是 data_types.FaceDetection 对象
            # 只需要确保 face_box 存在即可认为是一个有效检测
            if hasattr(face, 'face_box') and face.face_box:
                # 发布人脸检测事件，各引擎将异步处理
                self.event_bus.publish(
                    EventType.FACE_DETECTED,
                    {
                        'frame': frame.copy(), # 传递帧的副本
                        'face': face, # 直接传递 face_engine 返回的 face 对象
                        'frame_id': self.frame_count
                    }
                )
            else:
                # 记录一个更明确的错误，如果连 face_box 都没有
                self.logger.error(f"检测到无效的人脸对象（缺少 face_box）: {face}")

        # 更新UI (UI面板应监听各自需要的事件来获取最新数据)
        self.ui.update(frame, frame_result, self.fps)
        self.ui.render()
        self.ui.show()

def main():
    """
    模块直接运行的入口点
    """
    import argparse
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强版隐藏情绪检测系统')
    
    parser.add_argument('--source', type=str, default='0',
                        help='视频源，可以是摄像头索引或视频文件路径，默认为0（第一个摄像头）')
    
    parser.add_argument('--width', type=int, default=1280,
                        help='窗口宽度，默认为1280')
    
    parser.add_argument('--height', type=int, default=720,
                        help='窗口高度，默认为720')
    
    parser.add_argument('--models-dir', type=str, default=None,
                        help='模型文件目录，默认为enhance_hidden/models')
    
    parser.add_argument('--fps-limit', type=int, default=30,
                        help='帧率限制，默认为30fps')
    
    args = parser.parse_args()
    
    # 处理视频源
    if args.source.isdigit():
        args.source = int(args.source)
    
    # 创建应用实例
    app = HiddenEmotionApp(
        video_source=args.source,
        width=args.width,
        height=args.height,
        models_dir=args.models_dir,
        fps_limit=args.fps_limit
    )
    
    # 启动应用
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()
    except Exception as e:
        logging.error(f"应用运行时出错: {e}")
        import traceback
        traceback.print_exc()
        app.stop()

if __name__ == "__main__":
    main()