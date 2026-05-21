import numpy as np
from collections import defaultdict, deque

class Track:
    """单个目标跟踪类"""
    def __init__(self, track_id, bbox, frame_idx):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.history = deque(maxlen=30)  # 轨迹历史
        self.history.append((frame_idx, self._get_center()))
        self.age = 1
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        
    def _get_center(self):
        """计算边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox, frame_idx):
        """更新跟踪状态"""
        self.bbox = bbox
        self.history.append((frame_idx, self._get_center()))
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
    
    def mark_missing(self):
        """标记目标丢失"""
        self.consecutive_invisible_count += 1
    
    def is_confirmed(self):
        """确认跟踪是否稳定"""
        return self.total_visible_count >= 3
    
    def to_tlbr(self):
        """返回边界框格式"""
        return self.bbox

class EnhancedTracker:
    """增强的多目标跟踪器"""
    
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_id = 0
        self.frame_idx = 0
        self.debug_mode = True
        
        tracking_config = config.get('tracking', {})
        self.max_age = tracking_config.get('max_age', 30)
        self.min_hits = tracking_config.get('min_hits', 3)
        self.iou_threshold = tracking_config.get('iou_threshold', 0.3)
    
    def update(self, detections, frame):
        """更新跟踪状态"""
        self.frame_idx += 1
        
        if self.debug_mode and self.frame_idx % 50 == 0:
            print(f"🎯 跟踪器 - 帧 {self.frame_idx}: 检测目标 {len(detections)}, 现有轨迹 {len(self.tracks)}")
        
        # 预测现有跟踪器的位置
        for track in self.tracks:
            track.mark_missing()
        
        # 关联检测和跟踪
        matched_indices = []
        if detections and self.tracks:
            iou_matrix = self._calculate_iou_matrix(detections, self.tracks)
            matched_indices = self._hungarian_assign(iou_matrix)
            
            # 更新匹配的跟踪器
            for det_idx, track_idx in matched_indices:
                if iou_matrix[det_idx][track_idx] >= self.iou_threshold:
                    self.tracks[track_idx].update(detections[det_idx]['bbox'], self.frame_idx)
        
        # 创建新跟踪器
        for det_idx, det in enumerate(detections):
            if not self._is_detection_matched(det_idx, matched_indices):
                new_track = Track(self.next_id, det['bbox'], self.frame_idx)
                self.tracks.append(new_track)
                if self.debug_mode:
                    print(f"🆕 创建新轨迹 {self.next_id}")
                self.next_id += 1
        
        # 清理丢失的跟踪器
        initial_count = len(self.tracks)
        self.tracks = [track for track in self.tracks 
                      if track.consecutive_invisible_count <= self.max_age]
        removed_count = initial_count - len(self.tracks)
        if self.debug_mode and removed_count > 0:
            print(f"🗑️  移除 {removed_count} 个丢失的轨迹")
        
        # 返回确认的跟踪
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'history': list(track.history)
                })
        
        return confirmed_tracks
    
    def _calculate_iou_matrix(self, detections, tracks):
        """计算IoU矩阵"""
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_matrix[i][j] = self._calculate_iou(det['bbox'], track.bbox)
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        xi1 = max(x1_2, x1)
        yi1 = max(y1_2, y1)
        xi2 = min(x2_2, x2)
        yi2 = min(y2_2, y2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集区域
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _hungarian_assign(self, cost_matrix):
        """使用匈牙利算法进行关联"""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 最大化IoU
            return list(zip(row_ind, col_ind))
        except ImportError:
            # 简单的贪婪匹配
            matched = []
            rows, cols = cost_matrix.shape
            for i in range(rows):
                max_j = -1
                max_iou = -1
                for j in range(cols):
                    if cost_matrix[i][j] > max_iou:
                        max_iou = cost_matrix[i][j]
                        max_j = j
                if max_j != -1 and max_iou >= self.iou_threshold:
                    matched.append((i, max_j))
            return matched
    
    def _is_detection_matched(self, det_idx, matched_indices):
        """检查检测是否已被匹配"""
        for matched_det_idx, _ in matched_indices:
            if det_idx == matched_det_idx:
                return True
        return False