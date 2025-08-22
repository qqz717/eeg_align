import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置（递归）"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置有效性"""
    required_sections = ['model', 'training', 'datasets', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置缺少必需的部分: {section}")
    
    # 验证模型配置
    model_config = config['model']
    required_model_keys = ['n_channels', 'd_model', 'sample_rate']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"模型配置缺少必需的键: {key}")
    
    # 验证训练配置
    training_config = config['training']
    required_training_keys = ['num_epochs', 'batch_size', 'learning_rate']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"训练配置缺少必需的键: {key}")
    
    return True