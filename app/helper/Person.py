from dataclasses import dataclass  # Import the dataclass decorator
from typing import Optional, List, Tuple  # Import required typing components

@dataclass
class Person:
    """Person object to track individual people and their movement"""
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    is_moving: bool = False
    current_center: Optional[Tuple[int, int]] = None
    frames_tracked: int = 0
    position_history: List[Tuple[int, int]] = None
    history_size: int = 15  # Track last 15 frames (about 0.5 seconds at 30fps)
    oven_bbox: Optional[Tuple[int, int, int, int]] = (1000, 500, 1250, 720)
    def __post_init__(self):
        if self.position_history is None:
            self.position_history = []
        self.current_center = self.get_center()
        self.position_history.append(self.current_center)
    
    def get_center(self) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def update_position(self, x1: int, y1: int, x2: int, y2: int, confidence: float, movement_threshold: int = 15):
        """Update person's position and calculate if moving based on position history"""
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence
        self.current_center = self.get_center()
        self.frames_tracked += 1
        
        # Add current position to history
        self.position_history.append(self.current_center)
        
        # Keep only recent history
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # Calculate movement based on position history
        self.is_moving = self._calculate_movement_from_history(movement_threshold)
        
        # # Debug: print movement info occasionally
        # if self.frames_tracked % 20 == 0:  # Print every 20 frames
        #     avg_movement = self._get_average_movement()
        #     print(f"Person {self.id}: avg_movement={avg_movement:.1f}, moving={self.is_moving}, history_len={len(self.position_history)}")
    
    def _calculate_movement_from_history(self, threshold: int) -> bool:
        """Calculate if person is moving based on position history"""
        if len(self.position_history) < 10:  # Need at least 3 positions
            return False
        
        # Method 1: Check total displacement over history
        total_displacement = self._get_total_displacement()
        if total_displacement > threshold * 2:  # More lenient for total displacement
            return True
        
        # Method 2: Check average frame-to-frame movement
        avg_movement = self._get_average_movement()
        if avg_movement > threshold / 2:  # More sensitive for average movement
            return True
        
        # Method 3: Check if current position is far from oldest position
        if len(self.position_history) >= self.history_size:
            oldest_pos = self.position_history[0]
            current_pos = self.position_history[-1]
            distance = np.sqrt((current_pos[0] - oldest_pos[0])**2 + 
                             (current_pos[1] - oldest_pos[1])**2)
            if distance > threshold * 3:  # Check over longer period
                return True
        
        return False
    
    def _get_total_displacement(self) -> float:
        """Calculate total displacement over position history"""
        if len(self.position_history) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.position_history)):
            prev_pos = self.position_history[i-1]
            curr_pos = self.position_history[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            total += distance
        
        return total
    
    def _get_average_movement(self) -> float:
        """Calculate average frame-to-frame movement"""
        if len(self.position_history) < 2:
            return 0.0
        
        total_distance = self._get_total_displacement()
        return total_distance / (len(self.position_history) - 1)
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def distance_to(self, other_center: Tuple[int, int]) -> float:
        """Calculate distance to another center point"""
        if self.current_center is None:
            return float('inf')
        return np.sqrt((self.current_center[0] - other_center[0])**2 + 
                      (self.current_center[1] - other_center[1])**2)

    def checkIfGoToOven(self, target_bbox: Tuple[int, int, int, int], 
                        frame_id: int, 
                        total_frames: int = 5, 
                        angle_tolerance: float = 15.0, 
                        size_tolerance: float = 1.5) -> bool:
        """
        Check if the person is moving toward a predefined oven location and stops near it.

        Args:
            target_bbox (Tuple[int, int, int, int]): The oven's bounding box (x1, y1, x2, y2).
            frame_id (int): The starting frame ID to consider the movement.
            total_frames (int): Number of consecutive frames to check for stopping near the oven.
            angle_tolerance (float): Allowed deviation in degrees for direction matching.
            size_tolerance (float): Tolerance factor based on the person's bounding box size.

        Returns:
            bool: True if the person moves toward and stops near the oven, False otherwise.
        """
        if len(self.position_history) < total_frames:
            return False

        # Calculate the center of the oven bounding box
        target_center = ((self.oven_bbox[0] + self.oven_bbox[2]) // 2, 
                         (self.oven_bbox[1] + self.oven_bbox[3]) // 2)
        
        # Determine the person's size (diagonal of bounding box)
        person_size = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

        # Consider position history only after the given frame ID
        history_since_frame = self.position_history[-(self.frames_tracked - frame_id):]

        # Check if the person is near the oven
        current_pos = history_since_frame[-1]
        distance_to_target = np.sqrt((current_pos[0] - target_center[0])**2 + 
                                     (current_pos[1] - target_center[1])**2)

        if distance_to_target > size_tolerance * person_size:
            return False

        # Check if the person moved toward the oven
        start_pos = history_since_frame[0]
        movement_vector = (current_pos[0] - start_pos[0], current_pos[1] - start_pos[1])
        target_vector = (target_center[0] - start_pos[0], target_center[1] - start_pos[1])
        
        dot_product = (movement_vector[0] * target_vector[0] + movement_vector[1] * target_vector[1])
        mag_movement = np.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
        mag_target = np.sqrt(target_vector[0]**2 + target_vector[1]**2)
        
        if mag_movement == 0 or mag_target == 0:  # Prevent division by zero
            return False
        
        cos_theta = dot_product / (mag_movement * mag_target)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        if angle > angle_tolerance:
            return False

        # Check if the person stopped near the oven for a few consecutive frames
        near_target_count = sum(
            np.sqrt((pos[0] - target_center[0])**2 + (pos[1] - target_center[1])**2) <= size_tolerance * person_size
            for pos in history_since_frame[-total_frames:]
        )

        return near_target_count == total_frames
