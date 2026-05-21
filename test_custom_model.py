#!/usr/bin/env python3
"""
测试自定义训练模型
"""

import cv2
import os
from models.detector import PigDetector
from utils.config_loader import Config
from utils.visualizer import Visualizer

def test_custom_model():
    """测试自定义模型"""
    
    # 使用自定义配置
    config = Config("configs/custom_model.yaml")
    
    # 初始化检测器和可视化器
    detector = PigDetector(config)
    visualizer = Visualizer(config)
    
    # 测试图片路径
    test_images = [
    "data/pig-photo/115_8_output_5389.jpg",
    "pig_counting_project/data/pig-photo/downloaded_20241105_100859----00000118.jpg",
    "data/pig-photo/115_8_output_5345.jpg"
]
    
    # 创建输出目录
    os.makedirs("output/model_test", exist_ok=True)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"⚠️ 测试图片不存在: {img_path}")
            continue
            
        print(f"🔍 测试图片: {img_path}")
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 无法读取图片: {img_path}")
            continue
        
        # 检测猪只
        detections = detector.detect(image)
        print(f"✅ 检测到 {len(detections)} 只猪")
        
        # 可视化结果
        vis_image = visualizer.draw_detections(image, detections)
        
        # 保存结果
        output_path = f"output/model_test/detected_{os.path.basename(img_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"💾 结果保存至: {output_path}")
        
        # 显示检测详情
        for i, det in enumerate(detections):
            print(f"  猪只 {i+1}: 置信度 {det['confidence']:.3f}, 类别: {det['class_name']}")

if __name__ == "__main__":
    test_custom_model()