import cv2
import os
from typing import Generator, Tuple, Optional, Any

class VideoProcessor:
    """视频处理类"""
    
    def __init__(self, video_path: str, output_path: Optional[str] = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        # 视频信息
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 输出设置
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 修正：VideoWriter_fourcc
            self.writer = cv2.VideoWriter(
                f"{output_path}.mp4", fourcc, self.fps, (self.width, self.height)
            )
    
    def __iter__(self) -> Generator[Tuple[int, Any], None, None]:  # 修正类型注解
        """迭代器"""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    
    def write_frame(self, frame):
        """写入帧"""
        if self.writer:
            self.writer.write(frame)
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
    
    def __del__(self):
        self.release()