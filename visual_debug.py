#!/usr/bin/env python3
"""
可视化调试工具
"""

import cv2
import sys
from models.detector import PigDetector
from models.tracker import EnhancedTracker
from utils.counter import Counter
from utils.config_loader import Config
from utils.visualizer import Visualizer

def visual_debug(video_path, output_path, config_path="configs/optimized_counting.yaml"):
    """可视化调试计数过程"""
    
    print("🎨 启动可视化调试...")
    
    # 初始化组件
    config = Config(config_path)
    detector = PigDetector(config)
    tracker = EnhancedTracker(config)
    counter = Counter(config)
    visualizer = Visualizer(config)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # 处理流程
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        count_info = counter.count(tracks, frame_idx)
        
        # 可视化
        vis_frame = visualizer.draw(frame, tracks, count_info, frame_idx)
        
        # 添加调试信息
        debug_text = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracks)}", 
            f"Total Count: {count_info['total_count']}",
            f"New Counts: {count_info.get('new_counts', 0)}"
        ]
        
        for i, text in enumerate(debug_text):
            y = 60 + i * 25
            cv2.putText(vis_frame, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 写入输出视频
        out.write(vis_frame)
        
        # 显示进度
        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"📊 进度: {frame_idx}/{total_frames} ({progress:.1f}%) - 计数: {count_info['total_count']}")
        
        # 可选：显示实时画面（会减慢处理速度）
        # cv2.imshow('Debug', vis_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ 可视化调试完成!")
    print(f"📁 输出文件: {output_path}")
    print(f"📊 最终计数: {count_info['total_count']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python visual_debug.py <输入视频> <输出视频> [配置路径]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "configs/optimized_counting.yaml"
    
    visual_debug(input_video, output_video, config_path)