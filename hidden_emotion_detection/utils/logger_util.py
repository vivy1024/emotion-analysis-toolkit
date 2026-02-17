import logging
import os

def get_module_logger(name, log_file):
    # 优先从环境变量获取日志目录
    log_dir = os.environ.get("RUN_LOG_DIR", "logs")
    log_file_name = os.path.basename(log_file)
    log_file_path = os.path.join(log_dir, log_file_name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 防止日志冒泡导致重复输出

    # 文件日志（全量）
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file_path) for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 控制台日志（只输出INFO及以上）
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 只输出INFO及以上
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger 