import os
from ultralytics import YOLO
import logging
import numpy as np
import time

class PigDetector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
        self.class_name = "pig"  # 自定义模型的类别名称
    
    def _load_model(self, retry_count=3):
        """加载自定义训练的YOLO模型"""
        detection_config = self.config.get('detection', {})
        model_path = detection_config.get('model', 'yolov8n.pt')
        
        # 处理Windows路径
        model_path = model_path.replace('\\', '/')
        
        print(f"🔄 正在加载模型: {model_path}")
        
        for attempt in range(retry_count):
            try:
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    print(f"❌ 模型文件不存在: {model_path}")
                    # 尝试使用相对路径
                    alt_path = os.path.join(os.path.dirname(__file__), model_path)
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"✅ 使用相对路径: {model_path}")
                    else:
                        raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
                # 加载模型
                model = YOLO(model_path)
                
                # 验证模型是否加载成功
                if hasattr(model, 'names'):
                    print(f"✅ 模型加载成功! 类别: {model.names}")
                else:
                    print("✅ 模型加载成功!")
                
                return model
                
            except Exception as e:
                print(f"❌ 模型加载失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    time.sleep(2)  # 等待后重试
                else:
                    print("🚨 所有重试都失败，使用备用模型")
                    # 使用备用模型
                    try:
                        return YOLO('yolov8n.pt')
                    except:
                        raise RuntimeError("无法加载任何模型")
    
    def detect(self, image):
        """使用自定义模型检测图像中的猪只"""
        try:
            detection_config = self.config.get('detection', {})
            confidence_threshold = detection_config.get('confidence', 0.5)
            iou_threshold = detection_config.get('iou_threshold', 0.5)
            
            # 使用模型进行预测
            results = self.model(
                image, 
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False  # 减少输出
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标和置信度
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # 获取类别名称
                        if hasattr(self.model, 'names') and cls_id in self.model.names:
                            class_name = self.model.names[cls_id]
                        else:
                            class_name = f"class_{cls_id}"
                        
                        # 只添加置信度高于阈值的检测结果
                        if conf >= confidence_threshold:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class_name': class_name,
                                'class_id': cls_id
                            })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            return []