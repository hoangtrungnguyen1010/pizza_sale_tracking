"""
Tracker implementations following SOLID principles.
Uses composition and strategy patterns for flexibility and maintainability.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Type
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

from app.core.base import TrackerInterface, BaseTrackedObject, BoundingBox, MatchingStrategy, DetectionStrategy
from app.core.objects import Pizza, Person
from app.config import config

logger = logging.getLogger(__name__)


class IoUMatching(MatchingStrategy):
    """
    IoU-based matching strategy.
    Uses Intersection over Union for matching objects.
    """
    
    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        return bbox1.iou(bbox2)
    
    def match(self, detections: List[Tuple[int, int, int, int, float]], 
              tracked_objects: List[BaseTrackedObject]) -> Dict[int, int]:
        """
        Match detections to tracked objects using IoU.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            tracked_objects: List of currently tracked objects
            
        Returns:
            Dictionary mapping detection index to object ID
        """
        if not detections or not tracked_objects:
            return {}
        
        matches = {}
        used_objects = set()
        
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            detection_bbox = BoundingBox(x1, y1, x2, y2)
            best_match_id = None
            best_iou = 0.0
            
            for obj in tracked_objects:
                if obj.object_id in used_objects:
                    continue
                
                iou = self.calculate_iou(detection_bbox, obj.bbox)
                threshold = config.tracker.iou_threshold
                
                if iou > threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = obj.object_id
            
            if best_match_id is not None:
                matches[det_idx] = best_match_id
                used_objects.add(best_match_id)
        
        return matches


class DistanceMatching(MatchingStrategy):
    """
    Distance-based matching strategy.
    Uses center point distance for matching objects.
    """
    
    def calculate_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between centers of two bounding boxes."""
        center1 = bbox1.center
        center2 = bbox2.center
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def match(self, detections: List[Tuple[int, int, int, int, float]], 
              tracked_objects: List[BaseTrackedObject]) -> Dict[int, int]:
        """
        Match detections to tracked objects using distance.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            tracked_objects: List of currently tracked objects
            
        Returns:
            Dictionary mapping detection index to object ID
        """
        if not detections or not tracked_objects:
            return {}
        
        matches = {}
        used_objects = set()
        
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            detection_bbox = BoundingBox(x1, y1, x2, y2)
            best_match_id = None
            best_distance = float('inf')
            
            for obj in tracked_objects:
                if obj.object_id in used_objects:
                    continue
                
                distance = self.calculate_distance(detection_bbox, obj.bbox)
                threshold = config.tracker.max_distance
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = obj.object_id
            
            if best_match_id is not None:
                matches[det_idx] = best_match_id
                used_objects.add(best_match_id)
        
        return matches


class HungarianMatchingStrategy(MatchingStrategy):
    """
    Hungarian algorithm-based matching strategy.
    Follows Strategy Pattern - can be easily replaced with other matching algorithms.
    """
    
    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        return bbox1.iou(bbox2)
    
    def calculate_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between centers of two bounding boxes."""
        center1 = bbox1.center
        center2 = bbox2.center
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def match(self, detections: List[Tuple[int, int, int, int, float]], 
              tracked_objects: List[BaseTrackedObject]) -> Dict[int, int]:
        """
        Match detections to tracked objects using Hungarian algorithm.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            tracked_objects: List of currently tracked objects
            
        Returns:
            Dictionary mapping detection index to object ID
        """
        if not detections or not tracked_objects:
            return {}
        
        # Create cost matrix
        cost_matrix = np.zeros((len(tracked_objects), len(detections)))
        
        for i, obj in enumerate(tracked_objects):
            for j, (x1, y1, x2, y2, conf) in enumerate(detections):
                detection_bbox = BoundingBox(x1, y1, x2, y2)
                detection_center = detection_bbox.center
                
                # Calculate distance based on object type
                if isinstance(obj, Pizza):
                    cost = obj.calculate_combined_distance(detection_bbox, detection_center, None)
                else:
                    # For Person objects, use simple distance
                    cost = obj.distance_to(detection_center)
                
                cost_matrix[i, j] = cost
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on threshold
        matches = {}
        for row, col in zip(row_indices, col_indices):
            cost = cost_matrix[row, col]
            obj = tracked_objects[row]
            
            # Use different thresholds for different object types
            threshold = config.tracker.iou_threshold if isinstance(obj, Pizza) else config.person.max_distance_threshold
            
            if cost < threshold:
                matches[col] = obj.object_id
        
        return matches


class SimpleDistanceMatchingStrategy(MatchingStrategy):
    """
    Simple distance-based matching strategy.
    Alternative to Hungarian algorithm for simpler scenarios.
    """
    
    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        return bbox1.iou(bbox2)
    
    def calculate_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between centers of two bounding boxes."""
        center1 = bbox1.center
        center2 = bbox2.center
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def match(self, detections: List[Tuple[int, int, int, int, float]], 
              tracked_objects: List[BaseTrackedObject]) -> Dict[int, int]:
        """
        Match detections to tracked objects using simple distance.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            tracked_objects: List of currently tracked objects
            
        Returns:
            Dictionary mapping detection index to object ID
        """
        matches = {}
        used_objects = set()
        
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            detection_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            best_match_id = None
            best_distance = float('inf')
            
            for obj in tracked_objects:
                if obj.object_id in used_objects:
                    continue
                
                distance = obj.distance_to(detection_center)
                threshold = (config.tracker.max_distance if isinstance(obj, Pizza) 
                           else config.person.max_distance_threshold)
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = obj.object_id
            
            if best_match_id is not None:
                matches[det_idx] = best_match_id
                used_objects.add(best_match_id)
        
        return matches


class BaseTracker(TrackerInterface):
    """
    Base tracker implementation with common functionality.
    Follows Template Method Pattern - subclasses implement specific behavior.
    """
    
    def __init__(self, object_class: Type[BaseTrackedObject], 
                 matching_strategy: Optional[MatchingStrategy] = None):
        """
        Initialize the tracker.
        
        Args:
            object_class: Class of objects to track (Pizza or Person)
            matching_strategy: Strategy for matching detections to objects
        """
        self.object_class = object_class
        self.matching_strategy = matching_strategy or HungarianMatchingStrategy()
        
        # Tracking state
        self.next_object_id = 1
        self.active_objects: Dict[int, BaseTrackedObject] = {}
        self.last_seen_objects: Dict[int, BaseTrackedObject] = {}
        
        # Statistics
        self.frame_count = 0
        self.total_created = 0
        self.total_removed = 0
        
        logger.info(f"Initialized {self.__class__.__name__} for {object_class.__name__}")
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame: Optional[np.ndarray] = None) -> Dict[int, Any]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            frame: Current video frame (optional)
            
        Returns:
            Dictionary of active object positions
        """
        self.frame_count += 1
        
        # Handle empty detections
        if not detections:
            self._handle_no_detections()
            return self._get_active_objects_dict()
        
        # Match detections to existing objects
        matches = self.matching_strategy.match(detections, list(self.active_objects.values()))
        
        # Update matched objects
        self._update_matched_objects(detections, matches, frame)
        
        # Handle unmatched objects
        self._handle_unmatched_objects()
        
        # Process unmatched detections
        self._process_unmatched_detections(detections, matches)
        
        # Cleanup old objects
        self._cleanup_old_objects()
        
        return self._get_active_objects_dict()
    
    def get_active_objects(self) -> Dict[int, BaseTrackedObject]:
        """Get all currently active tracked objects."""
        return self.active_objects.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'frame_count': self.frame_count,
            'active_objects': len(self.active_objects),
            'last_seen_objects': len(self.last_seen_objects),
            'total_created': self.total_created,
            'total_removed': self.total_removed,
            'object_type': self.object_class.__name__
        }
    
    def _update_matched_objects(self, detections: List[Tuple[int, int, int, int, float]], 
                               matches: Dict[int, int], frame: Optional[np.ndarray]) -> None:
        """Update objects that were matched to detections."""
        for det_idx, object_id in matches.items():
            if object_id in self.active_objects:
                obj = self.active_objects[object_id]
                bbox, confidence = detections[det_idx][:4], detections[det_idx][4]
                
                # Use object-specific update method
                if isinstance(obj, Pizza):
                    obj.update_position(bbox, confidence, frame)
                else:
                    obj.update_position(bbox, confidence)
    
    def _handle_unmatched_objects(self) -> None:
        """Handle objects that weren't matched to any detection."""
        objects_to_remove = []
        
        for obj_id, obj in self.active_objects.items():
            obj.handle_disappearance()
            
            if obj.should_be_removed():
                objects_to_remove.append(obj_id)
        
        # Remove objects that should be removed
        for obj_id in objects_to_remove:
            self._remove_object(obj_id, "disappeared_too_long")
    
    def _process_unmatched_detections(self, detections: List[Tuple[int, int, int, int, float]], 
                                     matches: Dict[int, int]) -> None:
        """Process detections that weren't matched to any object."""
        for det_idx, (x1, y1, x2, y2, confidence) in enumerate(detections):
            if det_idx not in matches:
                self._create_new_object((x1, y1, x2, y2), confidence)
    
    def _create_new_object(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Create a new tracked object."""
        bbox_obj = BoundingBox(*bbox)
        
        if self.object_class == Pizza:
            obj = Pizza(object_id=self.next_object_id, bbox=bbox_obj, confidence=confidence)
        else:
            obj = Person(object_id=self.next_object_id, bbox=bbox_obj, confidence=confidence)
        
        self.active_objects[self.next_object_id] = obj
        self.next_object_id += 1
        self.total_created += 1
        
        logger.info(f"Created new {self.object_class.__name__} with ID {obj.object_id}")
    
    def _remove_object(self, object_id: int, reason: str) -> None:
        """Remove an object from active tracking."""
        if object_id in self.active_objects:
            obj = self.active_objects[object_id]
            self.last_seen_objects[object_id] = obj
            del self.active_objects[object_id]
            self.total_removed += 1
            
            logger.info(f"Removed {self.object_class.__name__} {object_id} - Reason: {reason}")
    
    def _handle_no_detections(self) -> None:
        """Handle the case when no detections are provided."""
        self._handle_unmatched_objects()
    
    def _cleanup_old_objects(self) -> None:
        """Remove very old objects from last_seen."""
        objects_to_cleanup = [
            obj_id for obj_id, obj in self.last_seen_objects.items()
            if obj.disappeared_count > config.tracker.max_disappeared * 2
        ]
        
        for obj_id in objects_to_cleanup:
            del self.last_seen_objects[obj_id]
    
    def _get_active_objects_dict(self) -> Dict[int, Any]:
        """Get dictionary of active object positions."""
        return {obj_id: obj.get_status_dict() for obj_id, obj in self.active_objects.items()}


class PizzaTracker(BaseTracker):
    """
    Specialized tracker for pizza objects.
    Extends BaseTracker with pizza-specific functionality.
    """
    
    def __init__(self, matching_strategy: Optional[MatchingStrategy] = None):
        """Initialize pizza tracker."""
        super().__init__(Pizza, matching_strategy)
        
        # Pizza-specific statistics
        self.total_sales = 0
        self.total_baked = 0
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame: Optional[np.ndarray] = None,
               people: Optional[List[Person]] = None) -> Dict[int, Any]:
        """
        Update pizza tracker with new detections and people information.
        
        Args:
            detections: List of pizza detections
            frame: Current video frame
            people: List of detected people for staff interaction
            
        Returns:
            Dictionary of active pizza positions
        """
        # Update staff interactions for all active pizzas
        if people:
            self._update_staff_interactions(people)
        
        # Call parent update method
        return super().update(detections, frame)
    
    def _update_staff_interactions(self, people: List[Person]) -> None:
        """Update staff interaction information for all active pizzas."""
        for pizza in self.active_objects.values():
            if isinstance(pizza, Pizza):
                pizza.update_staff_interactions(people, self.frame_count)
    
    def _remove_object(self, object_id: int, reason: str) -> None:
        """Override to handle pizza-specific removal logic."""
        if object_id in self.active_objects:
            pizza = self.active_objects[object_id]
            
            # Check if pizza went to oven using improved logic
            if isinstance(pizza, Pizza) and pizza.check_if_went_to_oven(self.frame_count):
                self.total_baked += 1
                reason = "baked"
                print(f"🍕 Pizza {object_id} was baked!")
            else:
                self.total_sales += 1
                reason = "sold"
                print(f"💰 Pizza {object_id} was sold!")
        
        super()._remove_object(object_id, reason)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pizza-specific statistics."""
        base_stats = super().get_statistics()
        base_stats.update({
            'total_sales': self.total_sales,
            'total_baked': self.total_baked
        })
        return base_stats


class PersonTracker(BaseTracker):
    """
    Specialized tracker for person objects.
    Extends BaseTracker with person-specific functionality.
    """
    
    def __init__(self, matching_strategy: Optional[MatchingStrategy] = None):
        """Initialize person tracker."""
        super().__init__(Person, matching_strategy)
    
    def _create_new_object(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Create a new person object with movement tracking."""
        bbox_obj = BoundingBox(*bbox)
        person = Person(object_id=self.next_object_id, bbox=bbox_obj, confidence=confidence)
        
        self.active_objects[self.next_object_id] = person
        self.next_object_id += 1
        self.total_created += 1
        
        logger.info(f"Created new Person with ID {person.object_id}")
    
    def get_moving_people(self) -> List[Person]:
        """Get all people that are currently moving."""
        return [person for person in self.active_objects.values() 
                if isinstance(person, Person) and person.is_moving]
    
    def get_stationary_people(self) -> List[Person]:
        """Get all people that are currently stationary."""
        return [person for person in self.active_objects.values() 
                if isinstance(person, Person) and not person.is_moving] 