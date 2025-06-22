"""
Centralized configuration for the pizza tracking application.
All configuration parameters are defined here to eliminate hardcoding.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
import os
import copy

@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    # YOLO model configuration
    model_path: str = "yolo11n.pt"
    pizza_class_id: int = 53
    
    # Confidence thresholds
    min_pizza_confidence: float = 0.3
    min_person_confidence: float = 0.3
    min_pizza_confidence_stationary: float = 0.15
    
    # Detection parameters
    expand_bbox_ratio: float = 0.1
    
    def __post_init__(self):
        if not isinstance(self.model_path, str):
            raise TypeError("model_path must be a string")
        if not isinstance(self.pizza_class_id, int):
            raise TypeError("pizza_class_id must be an int")
        for name in ["min_pizza_confidence", "min_person_confidence", "min_pizza_confidence_stationary", "expand_bbox_ratio"]:
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be a float")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'DetectionConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class TrackerConfig:
    """Configuration for object tracking"""
    # Disappearance tracking
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
    
    def __post_init__(self):
        for name in ["max_disappeared", "max_distance", "reappear_distance_threshold", "history_length", "new_object_check_frames", "established_object_threshold", "established_object_check_interval"]:
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ["pixel_similarity_threshold", "reappear_pixel_threshold", "iou_threshold", "iou_weight", "centroid_weight", "pixel_weight"]:
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be a float")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not isinstance(self.standard_pixel_size, tuple):
            raise TypeError("standard_pixel_size must be a tuple")
        if len(self.standard_pixel_size) != 2:
            raise ValueError("standard_pixel_size must be a tuple of length 2")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackerConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'TrackerConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class PersonConfig:
    """Configuration for person tracking"""
    # Movement detection
    movement_threshold: int = 15
    history_size: int = 15  # Track last 15 frames (about 0.5 seconds at 30fps)
    
    # Person tracker
    max_distance_threshold: int = 150
    
    # Pizza proximity detection
    pizza_proximity_threshold: float = 2.0  # Distance as multiple of pizza size
    
    # Staff visit tracking
    staff_visit_interval: int = 30  # Minimum frames between recording same staff visit
    
    # Oven detection
    oven_bbox: Tuple[int, int, int, int] = (1000, 500, 1250, 720)
    oven_angle_tolerance: float = 15.0
    oven_size_tolerance: float = 1.5
    oven_check_frames: int = 5
    
    # Oven visit timing
    min_oven_check_frames: int = 10  # Minimum frames after visit to check oven
    max_oven_check_frames: int = 300  # Maximum frames after visit to check oven
    
    def __post_init__(self):
        for name in ["movement_threshold", "history_size", "max_distance_threshold", "staff_visit_interval", "oven_check_frames", "min_oven_check_frames", "max_oven_check_frames"]:
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ["pizza_proximity_threshold", "oven_angle_tolerance", "oven_size_tolerance"]:
            value = getattr(self, name)
            if not isinstance(value, float):
                raise TypeError(f"{name} must be a float")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0")
        if not isinstance(self.oven_bbox, tuple):
            raise TypeError("oven_bbox must be a tuple")
        if len(self.oven_bbox) != 4:
            raise ValueError("oven_bbox must be a tuple of length 4")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'PersonConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class ImageProcessingConfig:
    """Configuration for image processing and enhancement"""
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # Brightness enhancement
    brightness_alpha: float = 1.1
    brightness_beta: int = 15
    
    # Pixel similarity weights
    ssim_weight: float = 0.5
    histogram_weight: float = 0.3
    mse_weight: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageProcessingConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'ImageProcessingConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class DisplayConfig:
    """Configuration for display and visualization"""
    # Font settings
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Colors (BGR format)
    pizza_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    person_color: Tuple[int, int, int] = (255, 0, 0)  # Red
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # Overlay settings
    overlay_alpha: float = 0.6
    overlay_beta: float = 0.4
    
    # Text positioning
    text_margin: int = 10
    text_spacing: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisplayConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'DisplayConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class VideoConfig:
    """Configuration for video processing"""
    # Video settings
    fps: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Processing settings
    scale_factor: float = 1.0
    enable_enhancement: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'VideoConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

@dataclass
class AppConfig:
    """Main application configuration"""
    # Flask settings
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 5000
    
    # File paths
    upload_folder: str = "uploads"
    output_folder: str = "output"
    temp_folder: str = "temp"
    
    # Allowed file extensions
    allowed_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Max file size (in bytes)
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def copy(self) -> 'AppConfig':
        """Create a copy of this config."""
        return copy.deepcopy(self)

class Config:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.tracker = TrackerConfig()
        self.person = PersonConfig()
        self.image_processing = ImageProcessingConfig()
        self.display = DisplayConfig()
        self.video = VideoConfig()
        self.app = AppConfig()
        
        # Load environment variables if they exist
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Detection config
        min_pizza_conf = os.getenv('MIN_PIZZA_CONFIDENCE')
        if min_pizza_conf is not None:
            self.detection.min_pizza_confidence = float(min_pizza_conf)
        
        min_person_conf = os.getenv('MIN_PERSON_CONFIDENCE')
        if min_person_conf is not None:
            self.detection.min_person_confidence = float(min_person_conf)
        
        # Tracker config
        max_disappeared = os.getenv('MAX_DISAPPEARED')
        if max_disappeared is not None:
            self.tracker.max_disappeared = int(max_disappeared)
        
        max_distance = os.getenv('MAX_DISTANCE')
        if max_distance is not None:
            self.tracker.max_distance = int(max_distance)
        
        # Person config
        movement_threshold = os.getenv('MOVEMENT_THRESHOLD')
        if movement_threshold is not None:
            self.person.movement_threshold = int(movement_threshold)
        
        person_max_distance = os.getenv('PERSON_MAX_DISTANCE')
        if person_max_distance is not None:
            self.person.max_distance_threshold = int(person_max_distance)
        
        # App config
        flask_debug = os.getenv('FLASK_DEBUG')
        if flask_debug is not None:
            self.app.debug = flask_debug.lower() == 'true'
        
        flask_host = os.getenv('FLASK_HOST')
        if flask_host is not None:
            self.app.host = flask_host
        
        flask_port = os.getenv('FLASK_PORT')
        if flask_port is not None:
            self.app.port = int(flask_port)
    
    def get_tracker_config(self):
        """Get tracker configuration section."""
        return self.tracker
    
    def get_person_config(self):
        """Get person configuration section."""
        return self.person
    
    def get_detection_config(self):
        """Get detection configuration section."""
        return self.detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            'detection': self.detection.to_dict(),
            'tracker': self.tracker.to_dict(),
            'person': self.person.to_dict(),
            'image_processing': self.image_processing.to_dict(),
            'display': self.display.to_dict(),
            'video': self.video.to_dict(),
            'app': self.app.to_dict()
        }
    
    def copy(self) -> 'Config':
        """Create a copy of this config."""
        return copy.deepcopy(self)

# Global configuration instance
config = Config()

# Backward compatibility functions
def get_tracker_config():
    """Get tracker configuration (backward compatibility)."""
    return config.get_tracker_config()

def get_person_config():
    """Get person configuration (backward compatibility)."""
    return config.get_person_config()

def get_detection_config():
    """Get detection configuration (backward compatibility)."""
    return config.get_detection_config()

def get_config() -> Dict[str, Any]:
    """Get the entire configuration as a dictionary."""
    return config.to_dict() 