#!/usr/bin/env python3
"""
任务一：图片猪只检测 - 使用自定义训练模型
"""

import argparse
import os
import cv2
import json
from models.detector import PigDetector
from utils.config_loader import Config
from utils.visualizer import Visualizer

def detect_images(input_path, output_dir, config_path="configs/custom_model.yaml"):
    """使用自定义模型检测图片中的猪只"""
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"⚠️ 自定义配置文件不存在，使用默认配置")
        config_path = "configs/default.yaml"
    
    # 初始化组件
    config = Config(config_path)
    detector = PigDetector(config)
    visualizer = Visualizer(config)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入图片列表
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = {}
    total_pigs = 0
    
    for image_path in image_paths:
        print(f"🔄 处理图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            continue
        
        # 检测猪只
        detections = detector.detect(image)
        
        # 保存结果
        filename = os.path.basename(image_path)
        results[filename] = []
        
        for det in detections:
            results[filename].append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class_name': det['class_name'],
                'class_id': det.get('class_id', 0)
            })
        
        total_pigs += len(detections)
        
        # 可视化结果
        vis_image = visualizer.draw_detections(image, detections)
        
        # 保存可视化图片
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, vis_image)
        
        print(f"✅ 检测到 {len(detections)} 只猪，结果保存至: {output_path}")
    
    # 保存边界框坐标
    results_file = os.path.join(output_dir, "detection_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 检测统计:")
    print(f"总图片数: {len(image_paths)}")
    print(f"总猪只数: {total_pigs}")
    print(f"边界框坐标保存至: {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='任务一：图片猪只检测（使用自定义模型）')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入图片路径或图片文件夹')
    parser.add_argument('--output', type=str, default='output/detection',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='configs/custom_model.yaml',
                       help='配置文件路径，默认使用自定义模型')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    results = detect_images(args.input, args.output, args.config)
    
    print(f"\n✅ 检测完成!")

if __name__ == "__main__":
    main()