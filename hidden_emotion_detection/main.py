#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版隐藏情绪检测系统 - 主程序入口

这个模块提供了系统的主入口，处理命令行参数并启动应用程序。
"""

# --- 首先导入并配置警告过滤 ---
import os
import sys
import warnings

# 在导入任何其他模块前配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*experimental.*")

# 标准库导入
import argparse
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# --- 先导入 ConfigManager --- 
from hidden_emotion_detection.config import config_manager

# --- 全局日志记录器 ---
# 不再在 setup_logging 中返回，直接获取根 logger
logger = logging.getLogger() # 获取根 logger

def setup_logging(level=logging.INFO, log_dir='logs'):
    """
    设置日志系统，同时输出到控制台和带时间戳的文件
    
    Args:
        level: 控制台日志级别，默认为 INFO
        log_dir: 日志文件存放目录
    """
    # 设置根 logger 的级别为最低（DEBUG），以便所有消息都能被处理
    logger.setLevel(logging.DEBUG) 
    
    # --- 创建 Formatter ---
    # 文件日志详细格式
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_formatter = logging.Formatter(file_format)
    
    # 控制台简洁格式 - 不显示时间和模块名称
    console_format = '%(levelname)s: %(message)s'
    console_formatter = logging.Formatter(console_format)
    
    # --- 添加日志过滤器，只保留重要的日志 ---
    class AUEmotionFilter(logging.Filter):
        """过滤器：只保留AU相关、情绪分析相关的日志"""
        def filter(self, record):
            # 保留的模块列表
            keep_modules = [
                'enhance_hidden.app', # APP 更名为 enhance_hidden.app
                'AU_EMOTION', 'AU_ENGINE', 'MICRO_EMOTION_AU', 'MACRO_EMOTION_AU',  # AU相关
                'MICRO_EMOTION', 'MACRO_EMOTION',  # 情绪分析相关
            ]
            
            # 保留的日志内容关键词
            keep_keywords = [
                'AU辅助', 'AU分析', '情绪分析', '微表情', '宏观表情', 
                '建议', '缓存命中', '处理耗时', '分析完成', '情绪结果'
            ]
            
            # 1. 保留指定模块的所有日志
            if record.name in keep_modules:
                return True
                
            # 2. 检查日志内容是否包含关键词
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                for keyword in keep_keywords:
                    if keyword in record.msg:
                        return True
            
            # 3. 高级别日志始终保留
            if record.levelno >= logging.WARNING:
                return True
                
            # 4. 其他情况过滤掉
            return False
    
    # --- 控制台严格过滤器 ---
    class ConsoleFilter(logging.Filter):
        """控制台过滤器：只显示ERROR及以上级别，以及特定重要信息"""
        def filter(self, record):
            # 优先处理：显示来自 enhance_hidden.app 的所有 ERROR 消息
            if record.name == 'enhance_hidden.app' and record.levelno == logging.ERROR:
                return True
                
            # 1. 始终显示 CRITICAL
            if record.levelno >= logging.CRITICAL:
                return True
            
            # 2. 隐藏所有低于 WARNING 的日志 (DEBUG, INFO)
            if record.levelno < logging.WARNING:
                return False

            # 3. 对于 WARNING 及以上，默认显示
            return True
    
    # 创建过滤器实例
    au_emotion_filter = AUEmotionFilter()
    console_filter = ConsoleFilter()
    
    # --- 配置控制台 Handler (StreamHandler) ---
    console_handler = logging.StreamHandler(sys.stdout) # 输出到标准输出
    console_handler.setLevel(logging.DEBUG) # 控制台级别调整为 DEBUG
    console_handler.setFormatter(console_formatter)
    # console_handler.addFilter(console_filter)  # 暂时注释掉严格控制台过滤器
    
    # --- 配置带时间戳的日志文件 Handler (FileHandler) ---
    file_handler = None # Initialize
    if log_dir: # 只有在指定了日志目录时才尝试创建文件
        # 确保日志目录存在
        print(f"[Log Setup] Attempting to use log directory: {log_dir}")
        if not os.path.exists(log_dir):
            print(f"[Log Setup] Log directory '{log_dir}' does not exist. Attempting to create.")
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"[Log Setup] Log directory '{log_dir}' created or already exists.")
            except OSError as e:
                print(f"[Log Setup ERROR] Failed to create log directory '{log_dir}': {e}", file=sys.stderr)
                log_dir = None # 无法创建目录，则不记录到文件
        
        if log_dir: # 如果目录存在或创建成功
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"app_{timestamp}.log")
            print(f"[Log Setup] Attempting to create FileHandler for: {log_file}")
            try:
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.DEBUG) # 文件记录 DEBUG 及以上
                file_handler.setFormatter(file_formatter)
                file_handler.addFilter(au_emotion_filter)  # 添加过滤器
                print(f"[Log Setup] FileHandler created for {log_file}") # DEBUG PRINT
            except Exception as e:
                print(f"[Log Setup ERROR] Failed to create log file '{log_file}': {e}", file=sys.stderr)
                file_handler = None # 创建失败
    else:
        print("[Log Setup] No log directory specified or creation failed, file logging disabled.")

    # --- 清除根 logger 可能存在的旧 Handler (避免重复添加) ---
    # 在重新配置时尤其重要
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # --- 添加新的 Handlers 到根 logger ---
    logger.addHandler(console_handler)  # 启用控制台输出
    if file_handler:
        logger.addHandler(file_handler)
        print("[Log Setup] FileHandler added to logger.") # DEBUG PRINT
    else:
        print("[Log Setup] FileHandler was not added to logger.") # DEBUG PRINT
    
    # --- 重新强化警告抑制 ---
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # 设置第三方库的日志级别为ERROR (只显示错误)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("google").setLevel(logging.ERROR)
    logging.getLogger("numpy").setLevel(logging.ERROR)
    
    # 捕获并重定向stderr警告
    class WarningCatcher:
        def write(self, message):
            # 只记录到文件，不输出到控制台
            if file_handler and message.strip() and 'warning' in message.lower():
                logger.debug(f"Warning captured: {message.strip()}")
            # 总是返回，以防后续处理
            return 0
            
        def flush(self):
            pass
    
    # 替换stderr以捕获警告 (只在调试模式下不替换)
    if level > logging.DEBUG:
        sys.stderr = WarningCatcher()

def parse_arguments():
    """
    解析命令行参数
    """
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
    
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    return parser.parse_args()


def setup_environment(args):
    """
    设置运行环境
    """
    # 设置日志级别
    if args.debug:
        logger.debug("调试模式已启用，所有模块日志级别已设为DEBUG")
    
    # 处理模型目录
    if args.models_dir is None:
        # 使用默认模型目录
        base_dir = Path(__file__).parent
        args.models_dir = str(base_dir / 'models')
        logger.debug(f"使用默认模型目录: {args.models_dir}")
    
    # 确保模型目录存在
    if not os.path.exists(args.models_dir):
        logger.error(f"模型目录不存在: {args.models_dir}")
        sys.exit(1)
    
    logger.info(f"使用模型目录: {args.models_dir}")
    
    # 处理视频源
    try:
        if args.source.isdigit():
            args.source = int(args.source)
            logger.info(f"使用摄像头: {args.source}")
        else:
            if not os.path.exists(args.source):
                logger.error(f"视频文件不存在: {args.source}")
                sys.exit(1)
            logger.info(f"使用视频文件: {args.source}")
    except Exception as e:
        logger.error(f"处理视频源时出错: {e}")
        logger.debug(f"异常详情: {str(e)}", exc_info=True)
        sys.exit(1)
    
    return args


def main():
    """
    主函数
    """
    # --- 从配置获取日志级别 (用于控制台) --- 
    log_level_str = config_manager.get('system.log_level', 'INFO').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    # 控制台级别，文件级别固定为 DEBUG
    console_log_level = log_level_map.get(log_level_str, logging.INFO) 
    
    # --- 初始化日志系统 (使用配置中的级别作为控制台级别) ---
    setup_logging(level=console_log_level, log_dir='logs') # <<< 修改调用，传递目录
    
    # 使用全局 logger 记录
    logger.info(f"程序启动，控制台日志级别设置为: {logging.getLevelName(console_log_level)}, 文件日志级别: DEBUG") 
    logger.critical("--- Log Test: Logging setup complete. This message should appear in console and file. ---") # TEST LOG
    
    # 解析命令行参数
    args = parse_arguments()
    logger.debug(f"命令行参数: {args}") # DEBUG 级别记录参数
    
    # 设置环境
    args = setup_environment(args)
    
    app = None # 初始化 app 变量
    try:
        # 导入应用程序类
        logger.debug("导入应用程序类")
        from hidden_emotion_detection.app import HiddenEmotionApp
        
        # 创建应用程序实例
        logger.debug("创建应用程序实例")
        app = HiddenEmotionApp(
            video_source=args.source,
            width=args.width,
            height=args.height,
            models_dir=args.models_dir,
            fps_limit=args.fps_limit
        )
        
        # 启动应用程序
        logger.info("启动增强版隐藏情绪检测系统...")
        app.start()
        
    except ImportError as e:
        logger.error(f"导入应用程序类时出错: {e}")
        logger.debug("导入错误详情", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动应用程序时出错: {e}")
        logger.debug("异常详情", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        # ---- 移除 Profiling End ----
        # profiler.disable()
        # logger.info("应用程序已停止，正在生成性能分析报告...")
        # s = io.StringIO()
        # sortby = pstats.SortKey.CUMULATIVE # 按累计时间排序
        # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        # ps.print_stats(30) # 打印前 30 个最耗时的函数
        # print("\n---- cProfile 性能分析结果 (Top 30 Cumulative Time) ----")
        # print(s.getvalue())
        # print("---- cProfile 性能分析结果结束 ----\n")
        # ---- 移除 Profiling End ----
        
        # 确保即使出错也尝试停止 app (如果已创建)
        if app and app.running:
             logger.info("尝试停止应用程序...")
             app.stop() 
        
        sys.exit(1 if 'e' in locals() and isinstance(e, Exception) else 0) # 如果有异常则以非零状态退出


if __name__ == "__main__":
    main()