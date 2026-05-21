#!/usr/bin/env python3
"""
诊断计数器问题 - 修复版
"""

import cv2
import sys
import os
from models.detector import PigDetector
from models.tracker import EnhancedTracker
from utils.counter import Counter
from utils.config_loader import Config
from utils.visualizer import Visualizer

def debug_video_counting(video_path, config_path="configs/custom_model.yaml"):
    """调试视频计数过程"""
    
    print("🔍 开始诊断计数问题...")
    
    # 初始化组件
    config = Config(config_path)
    detector = PigDetector(config)
    tracker = EnhancedTracker(config)
    counter = Counter(config)
    visualizer = Visualizer(config)
    
    # 打开视频 - 修复帧数获取方式
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 修复：正确的帧数获取方式
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果获取的帧数不合理，设置一个默认值
    if total_frames <= 0:
        print("⚠️ 无法获取准确帧数，使用手动计数")
        # 手动计算帧数（如果需要准确计数）
        temp_cap = cv2.VideoCapture(video_path)
        manual_count = 0
        while True:
            ret, _ = temp_cap.read()
            if not ret:
                break
            manual_count += 1
        temp_cap.release()
        total_frames = manual_count
    
    print(f"📊 视频总帧数: {total_frames}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        print(f"\n--- 帧 {frame_idx} ---")
        
        # 1. 检测阶段
        detections = detector.detect(frame)
        print(f"🔍 检测到 {len(detections)} 个目标")
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            print(f"   目标 {i}: bbox={bbox}, conf={conf:.3f}")
        
        # 2. 跟踪阶段
        tracks = tracker.update(detections, frame)
        print(f"🎯 跟踪到 {len(tracks)} 个轨迹")
        
        for track in tracks:
            print(f"   轨迹 {track['track_id']}: bbox={track['bbox']}")
        
        # 3. 计数阶段
        count_info = counter.count(tracks, frame_idx)
        print(f"📈 计数结果: {count_info}")
        
        # 每50帧显示一次进度
        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"📊 进度: {frame_idx}/{total_frames} ({progress:.1f}%)")
            
        # 只处理前200帧进行诊断
        if frame_idx >= 200:
            print("🛑 诊断完成（前200帧）")
            break
    
    cap.release()
    
    # 最终报告
    print(f"\n📋 诊断总结:")
    print(f"处理帧数: {frame_idx}")
    print(f"最终计数: {count_info['total_count']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python debug_counter.py <视频路径> [配置路径]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/custom_model.yaml"
    
    debug_video_counting(video_path, config_path)