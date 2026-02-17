import os
import logging
from datetime import datetime

# 设置日志记录器 (获取根 logger)
logger = logging.getLogger()
# 清除已有的处理器，防止重复添加
if logger.hasHandlers():
    logger.handlers.clear()

# 定义 setup_logging 函数
def setup_logging(log_dir, experiment_name):
    """配置日志记录，将日志保存到文件并输出到控制台。"""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 移除旧的处理器 (如果再次调用)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close() # 关闭文件句柄
        
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建格式化器
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(log_format)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"日志系统初始化完成。日志文件保存在: {log_filename}")

# Emotion mapping (Including 'fear' and 'sadness', assuming 7 classes based on CASME II)
# Matching the map from 18_1 for consistency
MICRO_EMOTION_MAP = {
    'happiness': 0,
    'disgust': 1,
    'repression': 2,
    'surprise': 3,
    'fear': 4,
    'sadness': 5,
    'others': 6,
}
NUM_CLASSES = len(MICRO_EMOTION_MAP)
# Create reverse mapping for logging/debugging
EMOTION_IDX_TO_NAME = {v: k for k, v in MICRO_EMOTION_MAP.items()}


def get_image_path(subject, filename, frame_num, data_root):
    """Constructs the full path to a specific image frame, trying different structures."""
    # Format subject folder (e.g., sub01, sub12)
    try:
        # Attempt standard formatting first
        sub_folder = f"sub{int(subject):02d}"
    except (ValueError, TypeError):
        # Handle cases where subject might already be formatted or non-numeric
        if isinstance(subject, str) and subject.startswith("sub") and subject[3:].isdigit():
            sub_folder = subject
        elif isinstance(subject, str) and subject.isdigit():
             sub_folder = f"sub{int(subject):02d}"
        else:
             # Fallback: use subject directly (e.g., if it's 'EP02_01f')
             # This might happen with non-standard subject naming in metadata
            # logger.warning(f"Non-standard subject format encountered: {subject}. Using directly.")
            sub_folder = str(subject)

    # Format image filename (e.g., img001.jpg, img123.jpg)
    # Assuming frame_num is integer
    try:
        img_file = f"img{int(frame_num)}.jpg"
    except (ValueError, TypeError):
         # 使用根 logger 记录错误
         logging.error(f"Invalid frame number format: {frame_num} for subject={subject}, filename={filename}")
         img_file = f"img{frame_num}.jpg" # Try using original format as fallback


    # --- Paths to Try ---
    # Based on CASME II structure and potential variations seen previously
    paths_to_try = [
        # Standard: data_root / subXX / Filename / imgYYY.jpg
        os.path.join(data_root, sub_folder, filename, img_file),
        # Variation: data_root / Filename / imgYYY.jpg (if no subject subfolder)
        os.path.join(data_root, filename, img_file),
        # Variation: data_root / subXX / imgYYY.jpg (if no Filename subfolder)
        os.path.join(data_root, sub_folder, img_file),
         # Variation: Using raw subject string if formatting failed
        os.path.join(data_root, str(subject), filename, img_file),
    ]

    # Check which path exists
    for path in paths_to_try:
        if os.path.exists(path):
            # logger.debug(f"Found image at: {path}")
            return path

    # If none found, log a warning and return the primary expected path
    # logging.warning(f"Could not find image for subject={subject}, filename={filename}, frame={frame_num}. Tried: {paths_to_try}. Returning default guess: {paths_to_try[0]}")
    return paths_to_try[0] # Return the most likely path even if not found, caller should handle None/Error 