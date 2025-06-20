from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrackerConfig:
    """Configuration parameters for simple tracker"""
    max_disappeared: int = 30
    max_distance: int = 100
    pixel_similarity_threshold: float = 0.5
    reappear_distance_threshold: int = 120
    reappear_pixel_threshold: float = 0.6
    history_length: int = 5
    standard_pixel_size: Tuple[int, int] = (32, 32)
    
    # Tracking parameters
    iou_threshold: float = 0.5
    
    # Matching weights
    iou_weight: float = 0.5
    centroid_weight: float = 0.3
    pixel_weight: float = 0.2
    
    # New object tracking parameters
    new_object_check_frames: int = 10
    established_object_threshold: int = 30
    established_object_check_interval: int = 30
