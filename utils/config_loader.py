import yaml
import os
from typing import Dict, Any

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self.data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置"""
        if not os.path.exists(self.config_path):
            print(f"⚠️ 配置文件不存在: {self.config_path}，使用默认配置")
            return self._get_default_config()
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                # 确保配置不为None
                return config_data if config_data is not None else self._get_default_config()
        except Exception as e:
            print(f"配置文件加载错误: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'detection': {
                'model': 'yolov8n.pt',  # 使用v8避免下载问题
                'confidence': 0.5,
                'iou_threshold': 0.5,
                'image_size': 640,
                'enhancement': True
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'use_kalman': True
            },
            'counting': {
                'count_lines': [
                    {'start': [100, 300], 'end': [800, 300], 'direction': 'both'}
                ]
            },
            'performance': {
                'use_gpu': True,
                'frame_skip': 0,
                'resize_factor': 1.0
            }
        }
    
    def summary(self) -> str:
        """配置摘要"""
        detection = self.data.get('detection', {})
        tracking = self.data.get('tracking', {})
        performance = self.data.get('performance', {})
        
        return (f"检测: {detection.get('model', 'unknown')} | "
                f"跟踪: {tracking.get('max_age', 'unknown')}帧 | "
                f"GPU: {performance.get('use_gpu', False)}")
    
    def __getitem__(self, key):
        return self.data.get(key, {})
    
    def get(self, key, default=None):
        return self.data.get(key, default)