from collections import defaultdict, deque
import numpy as np

class Counter:
    """智能计数器，支持虚拟线和区域计数"""
    
    def __init__(self, config):
        self.config = config
        self.count_lines = self._setup_count_lines()
        self.count_history = defaultdict(lambda: deque(maxlen=30))
        self.total_count = 0
        self.tracked_ids = set()  # 已经计数的ID
        self.debug_mode = True  # 调试模式
        
        print(f"🔢 计数器初始化完成，计数线: {len(self.count_lines)}条")
        for i, line in enumerate(self.count_lines):
            print(f"    计数线 {i}: {line['start']} -> {line['end']}, 方向: {line['direction']}")
    
    def _setup_count_lines(self):
        """设置计数线 - 可配置多条线"""
        lines = []
        # 默认计数线配置
        default_lines = [{'start': [100, 300], 'end': [800, 300], 'direction': 'both'}]
        
        config_lines = self.config.get('counting', {}).get('count_lines', default_lines)
        for line_config in config_lines:
            lines.append({
                'start': tuple(line_config['start']),
                'end': tuple(line_config['end']),
                'direction': line_config.get('direction', 'both'),
                'count': 0,
                'counted_ids': set()  # 记录已经在该线计数的ID
            })
        return lines
    
    def count(self, tracks, frame_idx):
        """执行计数逻辑"""
        frame_count = 0
        new_counts = 0
        
        if self.debug_mode and frame_idx % 50 == 0:
            print(f"🔍 计数器调试 - 帧 {frame_idx}: 跟踪目标数: {len(tracks)}")
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center = self._get_center(bbox)
            
            # 更新轨迹历史
            if track_id not in self.count_history:
                self.count_history[track_id] = deque(maxlen=30)
            self.count_history[track_id].append((frame_idx, center))
            
            # 检查计数线穿越
            line_cross, direction = self._check_line_crossing_with_direction(track_id, center)
            
            if line_cross and track_id not in self.tracked_ids:
                # 根据方向决定加减
                if direction == 'normal':  # 正常方向：左下到右上
                    frame_count += 1
                    self.total_count += 1
                    new_counts += 1
                    if self.debug_mode:
                        print(f"✅ 轨迹 {track_id} 正常方向穿越计数线，总数+1，当前总数: {self.total_count}")
                elif direction == 'reverse':  # 反方向
                    frame_count -= 1
                    self.total_count -= 1
                    new_counts -= 1
                    if self.debug_mode:
                        print(f"🔄 轨迹 {track_id} 反方向穿越计数线，总数-1，当前总数: {self.total_count}")
                
                self.tracked_ids.add(track_id)
        
        if self.debug_mode and new_counts != 0:
            print(f"📈 帧 {frame_idx} 新增计数: {new_counts}, 总计数: {self.total_count}")
        
        return {
            'frame_count': frame_count,
            'total_count': self.total_count,
            'active_tracks': len(tracks),
            'new_counts': new_counts
        }
    
    def _get_center(self, bbox):
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)
    
    def _check_line_crossing_with_direction(self, track_id, current_pos):
        """检查是否穿越计数线并判断方向"""
        if track_id not in self.count_history or len(self.count_history[track_id]) < 2:
            return False, None
            
        # 获取历史位置
        history = self.count_history[track_id]
        prev_pos = history[-2][1] if len(history) >= 2 else history[-1][1]
        
        for line in self.count_lines:
            if track_id in line['counted_ids']:
                continue
                
            crossed, direction = self._is_crossing_line_with_direction(
                prev_pos, current_pos, line['start'], line['end']
            )
            
            if crossed:
                line['counted_ids'].add(track_id)
                line['count'] += 1
                return True, direction
                
        return False, None
    
    def _is_crossing_line_with_direction(self, p1, p2, line_start, line_end):
        """判断线段是否相交并返回方向"""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        # 快速排斥实验
        if max(p1[0], p2[0]) < min(line_start[0], line_end[0]) or \
           min(p1[0], p2[0]) > max(line_start[0], line_end[0]) or \
           max(p1[1], p2[1]) < min(line_start[1], line_end[1]) or \
           min(p1[1], p2[1]) > max(line_start[1], line_end[1]):
            return False, None
        
        # 跨立实验
        if cross(p1, p2, line_start) * cross(p1, p2, line_end) <= 0 and \
           cross(line_start, line_end, p1) * cross(line_start, line_end, p2) <= 0:
            
            # 判断方向：根据移动向量与线向量的关系
            line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
            move_vector = (p2[0] - p1[0], p2[1] - p1[1])
            
            # 计算叉积判断方向
            cross_product = line_vector[0] * move_vector[1] - line_vector[1] * move_vector[0]
            
            # 假设正常方向是从左下到右上
            if cross_product > 0:
                return True, 'normal'  # 正常方向
            else:
                return True, 'reverse'  # 反方向
        
        return False, None
    
    def reset(self):
        """重置计数器"""
        self.total_count = 0
        self.tracked_ids.clear()
        self.count_history.clear()
        for line in self.count_lines:
            line['count'] = 0
            line['counted_ids'].clear()
        print("🔁 计数器已重置")