#!/usr/bin/env python3
"""
调试脚本 - 用于在VSCode中调试检测器
"""

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.detector import PigDetector
from utils.config_loader import Config

def debug_detector():
    """调试检测器"""
    print("🔧 开始调试检测器...")
    
    # 初始化
    config = Config()
    detector = PigDetector(config)
    
    # 测试图像
    test_image = cv2.imread('data/test_image.jpg')
    if test_image is None:
        # 创建测试图像
        test_image = cv2.imread('data/input_videos/test_frame.jpg')
        if test_image is None:
            print("❌ 没有找到测试图像")
            return
    
    print(f"📐 图像尺寸: {test_image.shape}")
    
    # 检测
    detections = detector.detect(test_image)
    print(f"🎯 检测到 {len(detections)} 个目标")
    
    # 可视化
    for i, det in enumerate(detections):
        bbox = det['bbox']
        conf = det['confidence']
        print(f"  {i+1}. 置信度: {conf:.3f}, 位置: {bbox}")
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(test_image, f'{conf:.2f}', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('调试结果', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ 调试完成!")

if __name__ == "__main__":
    debug_detector()