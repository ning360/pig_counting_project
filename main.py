"""
猪只计数系统主程序
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.detector import PigDetector
from models.tracker import EnhancedTracker
from utils.counter import Counter  
from utils.visualizer import Visualizer  
from utils.video_processor import VideoProcessor
from utils.config_loader import Config

class PigCountingSystem:  
    """猪只计数系统主类"""
    
    def __init__(self, config_path="configs/default.yaml"):
        self.config = Config(config_path)
        self.detector = PigDetector(self.config)
        self.tracker = EnhancedTracker(self.config)
        self.counter = Counter(self.config)  # 修正：使用Counter
        self.visualizer = Visualizer(self.config)  # 修正：使用Visualizer
        
        print("🐷 猪只计数系统初始化完成!")
        print(f"📊 配置: {self.config.summary()}")
    
    def process_video(self, video_path, output_path=None):
        """处理视频流"""
        processor = VideoProcessor(video_path, output_path)
        results = []
        
        try:
            for frame_idx, frame in processor:
                if frame is None:
                    break
                    
                # 处理流程
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                count_info = self.counter.count(tracks, frame_idx)  # 修正：使用count方法
                
                # 可视化
                if output_path:
                    vis_frame = self.visualizer.draw(frame, tracks, count_info, frame_idx)
                    processor.write_frame(vis_frame)
                
                # 记录结果
                results.append({
                    'frame_idx': frame_idx,
                    'detections': len(detections),
                    'tracks': len(tracks),
                    'total_count': count_info['total_count'],
                    'frame_count': count_info['frame_count']
                })
                
                # 进度显示
                if frame_idx % 50 == 0:
                    self._print_progress(frame_idx, processor.total_frames, count_info)
            
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断处理")
        except Exception as e:
            print(f"❌ 处理错误: {e}")
        finally:
            processor.release()
            
        return self._generate_report(results, output_path)
    
    def _print_progress(self, current, total, count_info):
        """打印处理进度"""
        progress = (current / total) * 100
        print(f"📈 进度: {current}/{total} ({progress:.1f}%) | "
              f"总数: {count_info['total_count']} | "
              f"当前跟踪: {count_info['active_tracks']}")
    
    def _generate_report(self, results, output_path):  
        """生成报告"""
        if not results:
            return {
                'final_count': 0,
                'total_frames': 0
            }
        
        final_count = results[-1]['total_count'] if results else 0
        total_frames = len(results)
        
        print(f"\n📊 最终报告:")
        print(f"总计数: {final_count}")
        print(f"处理帧数: {total_frames}")
        
        return {
            'final_count': final_count,
            'total_frames': total_frames,
            'results': results
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='猪只视频计数系统')
    parser.add_argument('--video', type=str, required=True, 
                       help='输入视频路径')
    parser.add_argument('--output', type=str, default='output',
                       help='输出路径前缀')
    parser.add_argument('--config', type=str, default='configs/custom_model.yaml',  # 修改默认配置
                       help='配置文件路径，默认使用自定义模型')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在，如果不存在则使用默认配置
    if not os.path.exists(args.config):
        print(f"⚠️ 配置文件 {args.config} 不存在，使用默认配置")
        args.config = 'configs/default.yaml'
    
    # 初始化系统
    system = PigCountingSystem(args.config)
    
    # 处理视频
    print(f"🎬 开始处理视频: {args.video}")
    results = system.process_video(args.video, args.output)
    
    print(f"\n✅ 处理完成!")
    print(f"📊 最终计数: {results['final_count']}")
    print(f"🎞️ 处理帧数: {results['total_frames']}")

if __name__ == "__main__":
    main()