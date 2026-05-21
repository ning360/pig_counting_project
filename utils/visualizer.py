import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
    
    def draw(self, image, tracks, count_info, frame_count):
        """绘制跟踪结果和计数信息"""
        vis_frame = image.copy()
        
        # 绘制计数线
        self._draw_count_lines(vis_frame)
        
        # 绘制跟踪轨迹
        self._draw_tracks(vis_frame, tracks)
        
        # 绘制计数信息
        self._draw_count_info(vis_frame, count_info, frame_count)
        
        return vis_frame
    
    def draw_detections(self, image, detections):
        """绘制检测结果（用于任务一）"""
        vis_frame = image.copy()
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            confidence = det['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[i % len(self.colors)]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"Pig: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制统计信息
        cv2.putText(vis_frame, f"Total Pigs: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_frame
    
    def _draw_count_lines(self, vis_frame):
        """绘制计数线"""
        try:
            count_lines = getattr(self.config, 'count_lines', [])
            
            for line in count_lines:
                if isinstance(line, dict):
                    start = tuple(line.get('start', [0, 0]))
                    end = tuple(line.get('end', [100, 100]))
                    direction = line.get('direction', 'up')
                    name = line.get('name', 'line')
                    
                    color = (0, 255, 0)  # 绿色
                    thickness = 2
                    cv2.line(vis_frame, start, end, color, thickness)
                    
                    self._draw_direction_arrow(vis_frame, start, end, direction, name)
                    
        except Exception as e:
            print(f"⚠️ 绘制计数线时出错: {e}")
            h, w = vis_frame.shape[:2]
            cv2.line(vis_frame, (0, h//2), (w, h//2), (0, 255, 0), 2)
    
    def _draw_tracks(self, vis_frame, tracks):
        """绘制跟踪轨迹"""
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            history = track.get('history', [])
            
            # 获取颜色
            color = self.colors[track_id % len(self.colors)]
            
            # 绘制轨迹
            for i in range(1, len(history)):
                _, prev_pos = history[i-1]
                _, curr_pos = history[i]
                cv2.line(vis_frame, 
                        (int(prev_pos[0]), int(prev_pos[1])),
                        (int(curr_pos[0]), int(curr_pos[1])),
                        color, 2)
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制跟踪ID
            label = f"ID: {track_id}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_count_info(self, vis_frame, count_info, frame_count):
        """绘制计数信息"""
        try:
            h, w = vis_frame.shape[:2]
            
            # 创建信息面板背景
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
            
            # 信息文本
            info_texts = [
                f"Frame: {frame_count}",
                f"Total Count: {count_info.get('total_count', 0)}",
                f"Current Frame: {count_info.get('frame_count', 0)}",
                f"Active Tracks: {count_info.get('active_tracks', 0)}"
            ]
            
            # 绘制信息文本
            for i, text in enumerate(info_texts):
                y_position = 30 + i * 25
                cv2.putText(vis_frame, text, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"⚠️ 绘制计数信息时出错: {e}")
    
    def _draw_direction_arrow(self, vis_frame, start, end, direction, name):
        """绘制方向箭头"""
        try:
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            
            arrow_length = 20
            if direction == "up":
                arrow_end = (mid_x, mid_y - arrow_length)
            elif direction == "down":
                arrow_end = (mid_x, mid_y + arrow_length)
            elif direction == "left":
                arrow_end = (mid_x - arrow_length, mid_y)
            elif direction == "right":
                arrow_end = (mid_x + arrow_length, mid_y)
            elif direction == "both":
                cv2.arrowedLine(vis_frame, (mid_x, mid_y), (mid_x, mid_y - arrow_length), (0, 255, 255), 2)
                cv2.arrowedLine(vis_frame, (mid_x, mid_y), (mid_x, mid_y + arrow_length), (0, 255, 255), 2)
                return
            else:
                return
            
            cv2.arrowedLine(vis_frame, (mid_x, mid_y), arrow_end, (0, 255, 255), 2)
            
            cv2.putText(vis_frame, name, (mid_x + 10, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception as e:
            print(f"⚠️ 绘制方向箭头时出错: {e}")