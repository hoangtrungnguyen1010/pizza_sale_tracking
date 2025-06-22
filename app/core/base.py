"""
Core base classes and interfaces following SOLID principles.
This module defines the fundamental abstractions for the tracking system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Protocol
import numpy as np

# Use absolute import for config
from app.config import config


class Positionable(Protocol):
    """Protocol for objects that have a position in 2D space."""
    
    @property
    def position(self) -> Tuple[int, int, int, int]:
        """Get the bounding box position (x1, y1, x2, y2)."""
        ...
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the object."""
        ...
    
    def distance_to(self, other_center: Tuple[int, int]) -> float:
        """Calculate distance to another center point."""
        ...


class Trackable(Protocol):
    """Protocol for objects that can be tracked across frames."""
    
    @property
    def id(self) -> int:
        """Get the unique identifier of the object."""
        ...
    
    @property
    def confidence(self) -> float:
        """Get the confidence score of the detection."""
        ...
    
    def update_position(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Update the object's position and confidence."""
        ...
    
    def is_visible(self) -> bool:
        """Check if the object is currently visible."""
        ...


@dataclass
class BoundingBox:
    """
    Represents a bounding box with utility methods.
    Follows Single Responsibility Principle - only handles bounding box operations.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box: x1 < x2 and y1 < y2 required")
    
    @property
    def width(self) -> int:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple format (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array format [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def expand(self, ratio: float = 0.1) -> 'BoundingBox':
        """Expand the bounding box by a given ratio."""
        expand_x = int(self.width * ratio)
        expand_y = int(self.height * ratio)
        
        return BoundingBox(
            x1=max(0, self.x1 - expand_x),
            y1=max(0, self.y1 - expand_y),
            x2=self.x2 + expand_x,
            y2=self.y2 + expand_y
        )
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Calculate intersection with another bounding box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 < x2 and y1 < y2:
            return BoundingBox(x1, y1, x2, y2)
        return None
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Calculate union with another bounding box."""
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2)
        )
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class BaseTrackedObject:
    """
    Base class for all tracked objects.
    Follows Single Responsibility Principle - handles common tracking functionality.
    """
    object_id: int
    bbox: BoundingBox
    confidence: float = 1.0
    frames_tracked: int = 0
    disappeared_count: int = 0
    is_new_object: bool = True
    
    def __post_init__(self):
        """Initialize the tracked object."""
        self._history: List[Tuple[int, int]] = []
        self._add_to_history(self.bbox.center)
    
    @property
    def position(self) -> Tuple[int, int, int, int]:
        """Get the bounding box position."""
        return self.bbox.to_tuple()
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point."""
        return self.bbox.center
    
    @property
    def history(self) -> List[Tuple[int, int]]:
        """Get the position history."""
        return self._history.copy()
    
    def _add_to_history(self, center: Tuple[int, int], max_history: Optional[int] = None) -> None:
        """Add center point to history."""
        if max_history is None:
            max_history = config.tracker.history_length
            
        self._history.append(center)
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]
    
    def update_position(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Update the object's position and confidence."""
        self.bbox = BoundingBox(*bbox)
        self.confidence = confidence
        self.disappeared_count = 0
        self.frames_tracked += 1
        self._add_to_history(self.bbox.center)
        
        # Mark as established if tracked long enough
        if self.frames_tracked >= config.tracker.established_object_threshold:
            self.is_new_object = False
    
    def handle_disappearance(self) -> None:
        """Handle when the object disappears from view."""
        self.disappeared_count += 1
        self.frames_tracked += 1
    
    def is_visible(self) -> bool:
        """Check if the object is currently visible."""
        return self.disappeared_count == 0
    
    def should_be_removed(self) -> bool:
        """Check if the object should be removed due to long disappearance."""
        max_disappeared = (config.tracker.new_object_check_frames 
                          if self.is_new_object else config.tracker.max_disappeared)
        return self.disappeared_count > max_disappeared
    
    def predict_next_position(self) -> Tuple[int, int]:
        """Predict the next position based on movement history."""
        if len(self._history) < 2:
            return self.center
        
        # Calculate average velocity from recent history
        recent_positions = self._history[-3:] if len(self._history) >= 3 else self._history
        if len(recent_positions) < 2:
            return self.center
        
        # Calculate velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            velocity = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            velocities.append(velocity)
        
        if velocities:
            avg_velocity_x = sum(v[0] for v in velocities) / len(velocities)
            avg_velocity_y = sum(v[1] for v in velocities) / len(velocities)
            predicted_x = int(self.center[0] + avg_velocity_x)
            predicted_y = int(self.center[1] + avg_velocity_y)
            return (predicted_x, predicted_y)
        
        return self.center
    
    def distance_to(self, other_center: Tuple[int, int]) -> float:
        """Calculate distance to another center point."""
        return np.sqrt((self.center[0] - other_center[0])**2 + 
                      (self.center[1] - other_center[1])**2)
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as dictionary."""
        return {
            'object_id': self.object_id,
            'position': self.position,
            'confidence': self.confidence,
            'frames_tracked': self.frames_tracked,
            'disappeared_count': self.disappeared_count,
            'is_new_object': self.is_new_object
        }


class TrackerInterface(ABC):
    """
    Abstract interface for trackers.
    Follows Interface Segregation Principle - defines only essential tracking methods.
    """
    
    @abstractmethod
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame: Optional[np.ndarray] = None) -> Dict[int, Any]:
        """Update tracker with new detections."""
        pass
    
    @abstractmethod
    def get_active_objects(self) -> Dict[int, Any]:
        """Get all currently active tracked objects."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        pass


class DetectionStrategy(ABC):
    """
    Abstract strategy for object detection.
    Follows Strategy Pattern and Open/Closed Principle.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect objects in the given frame."""
        pass


class MatchingStrategy(ABC):
    """
    Abstract strategy for matching detections to tracked objects.
    Follows Strategy Pattern and Open/Closed Principle.
    """
    
    @abstractmethod
    def match(self, detections: List[Tuple[int, int, int, int, float]], 
              tracked_objects: List[BaseTrackedObject]) -> Dict[int, int]:
        """Match detections to tracked objects. Returns {detection_idx: object_id}."""
        pass 