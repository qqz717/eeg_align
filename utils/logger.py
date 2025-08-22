import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional


def setup_logger(config: Dict) -> logging.Logger:
    """设置日志记录器"""
    
    # 创建日志目录
    log_dir = config.get('log_dir', 'experiments/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('EEG-WaveletVQ')
    logger.setLevel(getattr(logging, config.get('log_level', 'INFO')))
    
    # 清除现有handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


def get_logger(name: str = 'EEG-WaveletVQ') -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name)