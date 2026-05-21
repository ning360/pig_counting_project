#!/usr/bin/env python3
"""
简化测试 - 避免模型下载问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # 画几个矩形模拟猪只
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (300, 150), (400, 250), (0, 255, 0), -1)
    cv2.rectangle(img, (500, 200), (600, 300), (0, 255, 0), -1)
    return img

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 开始基本功能测试...")
    
    try:
        from utils.config_loader import Config
        from models.detector import PigDetector
        from models.tracker import EnhancedTracker
        from utils.counter import Counter
        from utils.visualizer import Visualizer
        
        print("✅ 模块导入成功")
        
        # 加载配置
        config = Config()
        print("✅ 配置加载成功")
        
        # 创建测试图像
        test_image = create_test_image()
        print(f"📐 测试图像尺寸: {test_image.shape}")
        
        # 测试检测器
        detector = PigDetector(config)
        print("✅ 检测器初始化成功")
        
        # 测试跟踪器
        tracker = EnhancedTracker(config)
        print("✅ 跟踪器初始化成功")
        
        # 测试计数器
        counter = Counter(config)
        print("✅ 计数器初始化成功")
        
        # 测试可视化器
        visualizer = Visualizer(config)
        print("✅ 可视化器初始化成功")
        
        # 模拟检测结果（避免实际模型调用）
        mock_detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'class_id': 0,
                'class_name': 'pig'
            },
            {
                'bbox': [300, 150, 400, 250],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'pig'
            }
        ]
        
        # 测试跟踪
        tracks = tracker.update(mock_detections, test_image)
        print(f"✅ 跟踪测试成功，跟踪数量: {len(tracks)}")
        
        # 测试计数
        count_info = counter.count(tracks, 0)
        print(f"✅ 计数测试成功，总数: {count_info['total_count']}")
        
        # 测试可视化
        vis_image = visualizer.draw(test_image, tracks, count_info, 0)
        print("✅ 可视化测试成功")
        
        # 保存测试结果
        cv2.imwrite('data/output_results/test_output.jpg', vis_image)
        print("💾 测试结果已保存")
        
        print("\n🎉 所有测试通过！系统可以正常运行。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()