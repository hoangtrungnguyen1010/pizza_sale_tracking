"""
Specific tracked object implementations.
Follows Liskov Substitution Principle - Pizza and Person can be used anywhere BaseTrackedObject is expected.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from app.core.base import BaseTrackedObject, BoundingBox, Positionable
from app.config import config


@dataclass
class Pizza(BaseTrackedObject):
    """
    Represents a tracked pizza object.
    Extends BaseTrackedObject with pizza-specific functionality.
    """
    # Pizza-specific properties
    pixels: Optional[np.ndarray] = None
    last_pixel_update: int = 0
    last_visit_staff: Optional[Tuple[Any, int]] = None  # (Person, frame_id)
    staff_visit_history: List[Tuple[Any, int]] = field(default_factory=list)  # List of (Person, frame_id)
    max_staff_history: int = 5  # Keep last 5 staff visits
    
    def update_position(self, bbox: Tuple[int, int, int, int], confidence: float, 
                       frame: Optional[np.ndarray] = None) -> None:
        """Update pizza position and extract pixel data."""
        super().update_position(bbox, confidence)
        
        # Extract and store pixel data for similarity comparison
        if frame is not None:
            self.pixels = self._extract_pixels(frame)
            self.last_pixel_update = self.frames_tracked
    
    def _extract_pixels(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract and resize pizza pixels from frame."""
        if frame is None:
            return None
        
        try:
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(self.bbox.x1, w))
            y1 = max(0, min(self.bbox.y1, h))
            x2 = max(x1, min(self.bbox.x2, w))
            y2 = max(y1, min(self.bbox.y2, h))
            
            # Check if we have a valid region
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract and resize the pizza region
            pizza_patch = frame[y1:y2, x1:x2]
            if pizza_patch.size == 0:
                return None
            
            resized_patch = cv2.resize(pizza_patch, config.tracker.standard_pixel_size)
            return resized_patch
            
        except Exception as e:
            print(f"Error extracting pixels for pizza {self.object_id}: {e}")
            return None
    
    def calculate_pixel_similarity(self, other_pixels: np.ndarray) -> float:
        """Calculate similarity between this pizza's pixels and other pixels."""
        if self.pixels is None or other_pixels is None:
            return 0.0
        
        try:
            # Convert to grayscale for comparison
            def to_gray(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            img1_gray = to_gray(self.pixels)
            img2_gray = to_gray(other_pixels)
            
            # Calculate structural similarity
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_score = ssim(img1_gray, img2_gray)
            except ImportError:
                # Fallback to template matching
                result = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
                ssim_score = np.max(result)
            
            # Calculate histogram correlation
            try:
                hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
                hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            except Exception:
                hist_corr = 0.0
            
            # Calculate mean squared error
            mse = np.mean((img1_gray.astype(float) - img2_gray.astype(float)) ** 2)
            mse_score = 1 / (1 + mse / 255)
            
            # Weighted combination
            return (config.image_processing.ssim_weight * ssim_score + 
                    config.image_processing.histogram_weight * hist_corr + 
                    config.image_processing.mse_weight * mse_score)
                    
        except Exception as e:
            print(f"Error calculating pixel similarity for pizza {self.object_id}: {e}")
            return 0.0
    
    def calculate_combined_distance(self, detection_bbox: BoundingBox, 
                                  detection_centroid: Tuple[int, int],
                                  detection_pixels: Optional[np.ndarray]) -> float:
        """Calculate combined distance using IoU, centroid, and pixel similarity."""
        # IoU distance
        iou_score = self.bbox.iou(detection_bbox)
        iou_distance = 1 - iou_score
        
        # Centroid distance (normalized)
        centroid_distance = self.distance_to(detection_centroid)
        normalized_centroid_distance = min(1.0, centroid_distance / config.tracker.max_distance)
        
        # Pixel similarity distance
        pixel_distance = 1.0
        if detection_pixels is not None:
            similarity = self.calculate_pixel_similarity(detection_pixels)
            pixel_distance = 1 - similarity
        
        # Weighted combination
        combined_distance = (config.tracker.iou_weight * iou_distance +
                           config.tracker.centroid_weight * normalized_centroid_distance +
                           config.tracker.pixel_weight * pixel_distance)
        
        return combined_distance
    
    def is_person_near_pizza(self, person: 'Person', proximity_threshold: Optional[float] = None) -> bool:
        """
        Check if a person is near the pizza (not just overlapping).
        
        Args:
            person: Person object to check
            proximity_threshold: Distance threshold (as multiple of pizza size)
            
        Returns:
            True if person is near the pizza
        """
        if proximity_threshold is None:
            proximity_threshold = config.person.pizza_proximity_threshold
        
        # Calculate distance between pizza center and person center
        distance = self.distance_to(person.center)
        
        # Use pizza size as reference for proximity
        pizza_size = float(max(self.bbox.width, self.bbox.height))
        max_distance = proximity_threshold * pizza_size
        
        return distance <= max_distance
    
    def overlaps_with_person(self, person: 'Person') -> float:
        """Calculate the overlap area between this pizza and a person."""
        intersection = self.bbox.intersection(person.bbox)
        return intersection.area if intersection else 0.0
    
    def find_nearby_staff(self, people: List['Person'], 
                         proximity_threshold: Optional[float] = None) -> List[Tuple['Person', float]]:
        """
        Find all staff members near this pizza, ordered by proximity.
        
        Args:
            people: List of detected people
            proximity_threshold: Distance threshold for considering someone "near"
            
        Returns:
            List of (Person, distance) tuples, sorted by distance
        """
        nearby_staff = []
        
        for person in people:
            if self.is_person_near_pizza(person, proximity_threshold):
                distance = self.distance_to(person.center)
                nearby_staff.append((person, distance))
        
        # Sort by distance (closest first)
        nearby_staff.sort(key=lambda x: x[1])
        return nearby_staff
    
    def update_staff_interactions(self, people: List['Person'], current_frame: int) -> None:
        """
        Update staff interaction information for this pizza.
        Tracks both overlap and proximity-based interactions.
        """
        # Find nearby staff
        nearby_staff = self.find_nearby_staff(people)
        
        if nearby_staff:
            # Get the closest staff member
            closest_staff, distance = nearby_staff[0]
            
            # Check if this is a new interaction or significant movement
            should_record_visit = False
            
            if self.last_visit_staff is None:
                # First visit
                should_record_visit = True
            else:
                last_staff, last_frame = self.last_visit_staff
                
                # Record visit if:
                # 1. Different staff member
                # 2. Same staff but significant time has passed
                # 3. Staff moved significantly closer
                if (last_staff.object_id != closest_staff.object_id or
                    current_frame - last_frame > config.person.staff_visit_interval or
                    distance < self._get_last_staff_distance() * 0.7):  # 30% closer
                    should_record_visit = True
            
            if should_record_visit:
                # Update last visit
                self.last_visit_staff = (closest_staff, current_frame)
                
                # Add to history
                self.staff_visit_history.append((closest_staff, current_frame))
                
                # Keep only recent history
                if len(self.staff_visit_history) > self.max_staff_history:
                    self.staff_visit_history = self.staff_visit_history[-self.max_staff_history:]
                
                print(f"Pizza {self.object_id}: Staff {closest_staff.object_id} visited at frame {current_frame}")
    
    def _get_last_staff_distance(self) -> float:
        """Get the distance to the last visiting staff member."""
        if self.last_visit_staff is None:
            return float('inf')
        
        last_staff, _ = self.last_visit_staff
        if last_staff is None:
            return float('inf')
        return self.distance_to(last_staff.center)
    
    def check_if_went_to_oven(self, current_frame: int) -> bool:
        """
        Check if this pizza went to oven based on staff interactions.
        
        Args:
            current_frame: Current frame number for timing calculations
            
        Returns:
            True if pizza likely went to oven
        """
        if not self.staff_visit_history:
            return False
        
        # Check recent staff visits (last 3 visits)
        recent_visits = self.staff_visit_history[-3:]
        
        for staff, visit_frame in recent_visits:
            # Check if this staff member went to oven after visiting the pizza
            if self._check_staff_oven_visit(staff, visit_frame, current_frame):
                print(f"Pizza {self.object_id}: Staff {staff.object_id} went to oven after visit")
                return True
        
        return False
    
    def _check_staff_oven_visit(self, staff: 'Person', visit_frame: int, current_frame: int) -> bool:
        """
        Check if a staff member went to oven after visiting this pizza.
        
        Args:
            staff: Staff member to check
            visit_frame: Frame when staff visited the pizza
            current_frame: Current frame number
            
        Returns:
            True if staff went to oven after visiting
        """
        # Check if staff went to oven within reasonable time after visit
        frames_since_visit = current_frame - visit_frame
        
        # Only check if reasonable time has passed (not too soon, not too late)
        if frames_since_visit < config.person.min_oven_check_frames:
            return False
        
        if frames_since_visit > config.person.max_oven_check_frames:
            return False
        
        # Check if staff went to oven
        return staff.check_if_went_to_oven()
    
    def get_staff_interaction_info(self) -> Dict[str, Any]:
        """Get information about staff interactions with this pizza."""
        info = {
            'has_staff_visits': len(self.staff_visit_history) > 0,
            'total_staff_visits': len(self.staff_visit_history),
            'last_visit_frame': None,
            'last_visit_staff_id': None
        }
        
        if self.last_visit_staff:
            staff, frame = self.last_visit_staff
            info['last_visit_frame'] = frame
            info['last_visit_staff_id'] = staff.object_id
        
        return info
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get pizza status information as dictionary."""
        base_status = super().get_status_dict()
        base_status.update({
            'has_pixels': self.pixels is not None,
            'last_pixel_update': self.last_pixel_update,
            'has_staff_visit': self.last_visit_staff is not None,
            'staff_visits_count': len(self.staff_visit_history)
        })
        return base_status


@dataclass
class Person(BaseTrackedObject):
    """
    Represents a tracked person object.
    Extends BaseTrackedObject with person-specific functionality.
    """
    # Person-specific properties
    is_moving: bool = False
    history_size: Optional[int] = None
    oven_bbox: Optional[BoundingBox] = None
    
    def __post_init__(self):
        """Initialize person-specific properties."""
        super().__post_init__()
        
        # Set config values if not provided
        if self.history_size is None:
            self.history_size = config.person.history_size
        if self.oven_bbox is None:
            oven_coords = config.person.oven_bbox
            self.oven_bbox = BoundingBox(*oven_coords)
    
    def update_position(self, bbox: Tuple[int, int, int, int], confidence: float,
                       movement_threshold: Optional[int] = None) -> None:
        """Update person position and calculate movement."""
        super().update_position(bbox, confidence)
        
        # Calculate movement based on position history
        if movement_threshold is None:
            movement_threshold = config.person.movement_threshold
        
        self.is_moving = self._calculate_movement_from_history(movement_threshold)
    
    def _calculate_movement_from_history(self, threshold: int) -> bool:
        """Calculate if person is moving based on position history."""
        if len(self._history) < 10:
            return False
        
        # Method 1: Check total displacement over history
        total_displacement = self._get_total_displacement()
        if total_displacement > threshold * 2:
            return True
        
        # Method 2: Check average frame-to-frame movement
        avg_movement = self._get_average_movement()
        if avg_movement > threshold / 2:
            return True
        
        # Method 3: Check if current position is far from oldest position
        history_size = self.history_size or config.person.history_size
        if len(self._history) >= history_size:
            oldest_pos = self._history[0]
            current_pos = self._history[-1]
            distance = np.sqrt((current_pos[0] - oldest_pos[0])**2 + 
                             (current_pos[1] - oldest_pos[1])**2)
            if distance > threshold * 3:
                return True
        
        return False
    
    def _get_total_displacement(self) -> float:
        """Calculate total displacement over position history."""
        if len(self._history) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self._history)):
            prev_pos = self._history[i-1]
            curr_pos = self._history[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            total += distance
        
        return total
    
    def _get_average_movement(self) -> float:
        """Calculate average frame-to-frame movement."""
        if len(self._history) < 2:
            return 0.0
        
        total_distance = self._get_total_displacement()
        return total_distance / (len(self._history) - 1)
    
    def check_if_went_to_oven(self, total_frames: Optional[int] = None,
                             angle_tolerance: Optional[float] = None,
                             size_tolerance: Optional[float] = None) -> bool:
        """
        Check if the person moved toward and stopped near the oven.
        
        Args:
            total_frames: Number of consecutive frames to check for stopping near the oven
            angle_tolerance: Allowed deviation in degrees for direction matching
            size_tolerance: Tolerance factor based on the person's bounding box size
            
        Returns:
            True if the person moves toward and stops near the oven, False otherwise
        """
        # Use config defaults if not provided
        if total_frames is None:
            total_frames = config.person.oven_check_frames
        if angle_tolerance is None:
            angle_tolerance = config.person.oven_angle_tolerance
        if size_tolerance is None:
            size_tolerance = config.person.oven_size_tolerance
        
        if len(self._history) < total_frames:
            return False
        
        # Calculate the center of the oven
        oven_center = self.oven_bbox.center
        
        # Determine the person's size (diagonal of bounding box)
        person_size = np.sqrt(self.bbox.width**2 + self.bbox.height**2)
        
        # Check if the person is near the oven
        current_pos = self._history[-1]
        distance_to_oven = np.sqrt((current_pos[0] - oven_center[0])**2 + 
                                  (current_pos[1] - oven_center[1])**2)
        
        if distance_to_oven > size_tolerance * person_size:
            return False
        
        # Check if the person moved toward the oven
        start_pos = self._history[0]
        movement_vector = (current_pos[0] - start_pos[0], current_pos[1] - start_pos[1])
        target_vector = (oven_center[0] - start_pos[0], oven_center[1] - start_pos[1])
        
        # Calculate angle between movement and target vectors
        dot_product = (movement_vector[0] * target_vector[0] + 
                      movement_vector[1] * target_vector[1])
        mag_movement = np.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
        mag_target = np.sqrt(target_vector[0]**2 + target_vector[1]**2)
        
        if mag_movement == 0 or mag_target == 0:
            return False
        
        cos_theta = dot_product / (mag_movement * mag_target)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        if angle > angle_tolerance:
            return False
        
        # Check if the person stopped near the oven for consecutive frames
        near_oven_count = sum(
            np.sqrt((pos[0] - oven_center[0])**2 + (pos[1] - oven_center[1])**2) <= size_tolerance * person_size
            for pos in self._history[-total_frames:]
        )
        
        return near_oven_count == total_frames
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get person status information as dictionary."""
        base_status = super().get_status_dict()
        base_status.update({
            'is_moving': self.is_moving,
            'history_size': self.history_size,
            'has_oven_bbox': self.oven_bbox is not None
        })
        return base_status 