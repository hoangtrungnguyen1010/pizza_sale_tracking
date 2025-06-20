from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .TrackerConfig import TrackerConfig

@dataclass
class TrackedObject:
    """Simple tracked object data class"""
    object_id: int
    centroid: np.ndarray
    rect: Tuple[int, int, int, int]
    disappeared_count: int = 0
    history: List[np.ndarray] = field(default_factory=list)
    pixels: Optional[np.ndarray] = None
    last_pixel_update: int = 0
    confidence: float = 1.0
    frames_tracked: int = 0  # Total frames this object has been tracked
    last_check_frame: int = 0  # Last frame where we checked for this object
    is_new_object: bool = True  # Whether this is a newly created object
    
    def add_to_history(self, centroid: np.ndarray, max_history: int = 5):
        """Add centroid to history with size limit"""
        self.history.append(centroid)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
    
    def predict_next_position(self) -> np.ndarray:
        """Simple prediction based on movement history"""
        if len(self.history) < 2:
            return self.centroid
        
        # Calculate average velocity from recent history
        recent_positions = self.history[-3:] if len(self.history) >= 3 else self.history
        if len(recent_positions) < 2:
            return self.centroid
        
        # Calculate velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            velocity = recent_positions[i] - recent_positions[i-1]
            velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities, axis=0)
            predicted_position = self.centroid + avg_velocity
            return predicted_position
        
        return self.centroid
    
    def should_check_this_frame(self, current_frame: int, config: 'TrackerConfig') -> bool:
        """Determine if we should check for this object in the current frame"""
        if self.is_new_object:
            # Always check new objects every frame
            return True
        elif self.frames_tracked >= config.established_object_threshold:
            # For established objects, check every N frames
            return (current_frame - self.last_check_frame) >= config.established_object_check_interval
        else:
            # For objects in between, check every frame
            return True
