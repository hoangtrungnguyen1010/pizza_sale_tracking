from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from scipy.spatial import distance as dist
import logging
from .TrackerConfig import TrackerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
    else:
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    if len(box2) == 4:
        x1_2, y1_2, x2_2, y2_2 = box2
    else:
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

@dataclass
class TrackedObject:
    """Enhanced tracked object with self-contained functionality"""
    object_id: int
    centroid: np.ndarray
    rect: Tuple[int, int, int, int]
    disappeared_count: int = 0
    history: List[np.ndarray] = field(default_factory=list)
    pixels: Optional[np.ndarray] = None
    last_pixel_update: int = 0
    confidence: float = 1.0
    frames_tracked: int = 0
    last_check_frame: int = 0
    is_new_object: bool = True
    last_visit_staff: Optional[Tuple[Any, int]] = None  # (Person, frame_id)
    
    def add_to_history(self, centroid: np.ndarray, max_history: int = 5):
        """Add centroid to history with size limit"""
        self.history.append(centroid)
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            

    def should_check_this_frame(self, current_frame: int, config: TrackerConfig) -> bool:
        """Determine if we should check for this object in the current frame"""
        if self.is_new_object:
            return True
        else:
            return current_frame  % 100 == 0
    
    def extract_and_resize_pixels(self, frame: np.ndarray, config: TrackerConfig) -> Optional[np.ndarray]:
        """Extract and resize object pixels from frame"""
        if frame is None:
            return None
        
        # Ensure rect has exactly 4 values and convert to int
        rect_values = list(self.rect)
        if len(rect_values) != 4:
            logger.warning(f"Invalid rect format for object {self.object_id}: {self.rect}")
            return None
            
        startX, startY, endX, endY = [int(val) for val in rect_values[:4]]
        h, w = frame.shape[:2]
        
        # Ensure coordinates are valid
        startX = max(0, min(startX, w))
        startY = max(0, min(startY, h))
        endX = max(startX, min(endX, w))
        endY = max(startY, min(endY, h))
        
        # Check if we have a valid region
        if endX <= startX or endY <= startY:
            logger.warning(f"Invalid bounding box for object {self.object_id}: ({startX}, {startY}, {endX}, {endY})")
            return None
        
        object_patch = frame[startY:endY, startX:endX]
        
        if object_patch.size == 0:
            return None
        
        try:
            resized_patch = cv2.resize(object_patch, config.standard_pixel_size)
            return resized_patch
        except cv2.error as e:
            logger.warning(f"CV2 resize error for object {self.object_id}: {e}")
            return None
    
    def calculate_pixel_similarity(self, other_pixels: np.ndarray) -> float:
        """Calculate similarity between this object's pixels and other pixels"""
        if self.pixels is None or other_pixels is None:
            return 0.0
            
        def to_gray(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        img1_gray = to_gray(self.pixels)
        img2_gray = to_gray(other_pixels)
        
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(img1_gray, img2_gray)
        except ImportError:
            result = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
            ssim_score = np.max(result)
        
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        mse = np.mean((img1_gray.astype(float) - img2_gray.astype(float)) ** 2)
        mse_score = 1 / (1 + mse / 255)
        
        return 0.5 * ssim_score + 0.3 * hist_corr + 0.2 * mse_score
    
    def calculate_combined_distance(self, detection_bbox: np.ndarray, 
                                  detection_centroid: np.ndarray,
                                  detection_pixels: Optional[np.ndarray],
                                  config: TrackerConfig) -> float:
        """Calculate combined distance using IoU, centroid, and pixel similarity"""
        # Convert object rect to bbox format, ensuring we have 4 values
        rect_values = list(self.rect)[:4]  # Take only first 4 values
        obj_bbox = np.array([rect_values[0], rect_values[1], rect_values[2], rect_values[3]])
        
        # IoU distance
        iou_score = iou(obj_bbox, detection_bbox)
        iou_distance = 1 - iou_score
        
        # Centroid distance (normalized)
        centroid_distance = dist.euclidean(self.centroid, detection_centroid)
        normalized_centroid_distance = min(1.0, centroid_distance / config.max_distance)
        
        # Pixel similarity distance
        pixel_distance = 1.0
        if detection_pixels is not None:
            similarity = self.calculate_pixel_similarity(detection_pixels)
            pixel_distance = 1 - similarity
        
        # Weighted combination
        combined_distance = (config.iou_weight * iou_distance +
                           config.centroid_weight * normalized_centroid_distance +
                           config.pixel_weight * pixel_distance)
        
        return combined_distance
    
    def update_with_detection(self, bbox: np.ndarray, frame: Optional[np.ndarray], 
                            config: TrackerConfig, current_frame: int):
        """Update this object with a new detection"""
        centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        # Ensure we store exactly 4 values as integers
        rect = tuple(int(val) for val in bbox[:4])
        
        # Update object properties
        self.centroid = centroid
        self.rect = rect
        self.disappeared_count = 0
        self.last_pixel_update = 0
        self.last_check_frame = current_frame
        
        # **FIX: Always increment frames_tracked when object is successfully matched**
        self.frames_tracked += 1
        
        # Mark as established if it has been tracked long enough
        if self.frames_tracked >= config.established_object_threshold:
            self.is_new_object = False
        
        # Update history and pixels
        self.add_to_history(centroid, config.history_length)
        if frame is not None:
            self.pixels = self.extract_and_resize_pixels(frame, config)
        
        logger.info("Updated object %d at position %s (frames_tracked: %d)", 
                   self.object_id, rect, self.frames_tracked)
    
    def handle_disappeared(self, current_frame: int, config: TrackerConfig, should_check_this_frame,
                          frame: Optional[np.ndarray] = None) -> str:
        """
        Handle when this object wasn't detected in current frame
        Returns: 'keep', 'move_to_last_seen', or 'baked'
        """
        # **FIX: Always increment frames_tracked even when object disappears**
        # This ensures the counter keeps going regardless of detection status
        self.frames_tracked += 1
        # Only increment disappeared count if we should check this object in this frame
        if not should_check_this_frame:
            return 'keep'
        
        self.disappeared_count += 1
        self.last_check_frame = current_frame
        
        # **FIX: Update is_new_object status even when disappeared**
        if self.frames_tracked >= config.established_object_threshold:
            self.is_new_object = False
        
        print(f"Frame {current_frame}: Object {self.object_id} disappeared count: {self.disappeared_count} "
              f"(new_object: {self.is_new_object}, frames_tracked: {self.frames_tracked})")
        
        # Different thresholds for new vs established objects
        max_disappeared = config.new_object_check_frames if self.is_new_object else config.max_disappeared
        
        # Check if object has been missing too long
        if self.disappeared_count > max_disappeared:
            # Check if object went to oven (you'll need to implement this logic)
            if self.check_if_went_to_oven():
                return 'baked'
            else:
                return 'move_to_last_seen'
        
        # Update pixels if available (check if object is still visible at last known location)
        if frame is not None and self.pixels is not None:
            current_patch = self.extract_and_resize_pixels(frame, config)
            if current_patch is not None:
                similarity = self.calculate_pixel_similarity(current_patch)
                if similarity > config.pixel_similarity_threshold:
                    # Object might still be there, reset disappeared count
                    self.disappeared_count = max(0, self.disappeared_count - 1)
                    self.pixels = current_patch  # Update pixels
                    print(f"Frame {current_frame}: Object {self.object_id} pixel similarity suggests it's still there")
        
        return 'keep'
    
    def check_if_went_to_oven(self) -> bool:
        """Check if this object went to oven (implement your logic here)"""
        if self.last_visit_staff and self.last_visit_staff[0]:
            # You'll need to implement this method in your Person class
            # return self.last_visit_staff[0].checkIfGoToOven(self, target_bbox, self.last_visit_staff[1])
            pass
        return False
    
    def overlaps_with(self, person) -> float:
        """Calculate the overlap area between this object and a person"""
        rect_values = list(self.rect)[:4]  # Ensure we have exactly 4 values
        x1_obj, y1_obj, x2_obj, y2_obj = rect_values
        x1_person, y1_person, x2_person, y2_person = person.x1, person.y1, person.x2, person.y2
        
        # Calculate intersection coordinates
        inter_x1 = max(x1_obj, x1_person)
        inter_y1 = max(y1_obj, y1_person)
        inter_x2 = min(x2_obj, x2_person)
        inter_y2 = min(y2_obj, y2_person)
        
        # Compute intersection area
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return 0.0
    
    def find_most_overlapping_person(self, people: List) -> Optional[Any]:
        """Find the person with the most overlap with this object"""
        max_overlap = 0.0
        most_overlapping_person = None
        
        for person in people:
            overlap_area = self.overlaps_with(person)
            if overlap_area > max_overlap:
                max_overlap = overlap_area
                most_overlapping_person = person
        
        return most_overlapping_person
    
    def update_last_visit_staff(self, people: List, current_frame: int):
        """Update the last staff member who visited this object"""
        most_overlapping = self.find_most_overlapping_person(people)
        if most_overlapping:
            self.last_visit_staff = (most_overlapping, current_frame)
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as dictionary"""
        return {
            'disappeared_count': self.disappeared_count,
            'confidence': self.confidence,
            'position': self.rect,
            'frames_tracked': self.frames_tracked,
            'is_new_object': self.is_new_object,
            'last_check_frame': self.last_check_frame
        }
    
    @classmethod
    def create_new(cls, object_id: int, bbox: np.ndarray, frame: Optional[np.ndarray], 
                   config: TrackerConfig, current_frame: int) -> 'TrackedObject':
        """Factory method to create a new tracked object"""
        centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        # Ensure we store exactly 4 values as integers
        rect = tuple(int(val) for val in bbox[:4])
        
        obj = cls(
            object_id=object_id,
            centroid=centroid,
            rect=rect,
            frames_tracked=1,  # Start with 1 since this is the first frame
            last_check_frame=current_frame,
            is_new_object=True
        )
        
        # Extract pixels
        if frame is not None:
            obj.pixels = obj.extract_and_resize_pixels(frame, config)
        
        obj.add_to_history(centroid, config.history_length)
        
        logger.info("Created new object with ID: %d at position %s", object_id, rect)
        print(f"✨ Created new object with ID: {object_id} at position {rect}")
        
        return obj

class SimpleTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        
        # Core tracking data
        self.next_object_id = 0
        self.active_objects: Dict[int, TrackedObject] = {}
        self.last_seen_objects: Dict[int, TrackedObject] = {}
        
        # Statistics
        self.total_sales = 0
        self.frame_count = 0
        self.total_baked = 0
        
        # Current frame for pixel comparison
        self.current_frame: Optional[np.ndarray] = None
        
        logger.info("Simple Tracker initialized")

    
    def should_check_this_frame(self) -> bool:
        """Determine if we should check for this object in the current frame"""
        return self.frame_count  % 100 == 0

    def update(self, detections: Optional[List[Tuple[int, int, int, int]]] = None, 
               frame: Optional[np.ndarray] = None, staff_tracker=None) -> Dict[int, np.ndarray]:
        """Update tracker with new detections"""
        self.current_frame = frame
        self.frame_count += 1
        
        # Update staff information for all active objects
        if staff_tracker and hasattr(staff_tracker, 'people'):
            self._update_staff_visits(staff_tracker.people)
        
        # Handle empty detections
        if detections is None or len(detections) == 0:
            self._handle_no_detections()
            return self._get_active_objects_centroids()
        
        detections_array = np.array(detections, dtype=np.float32)
        unmatched_detections = set(range(len(detections_array)))
        
        if len(self.active_objects) > 0:
            unmatched_detections = self._match_detections_to_active_objects(detections_array)
        
        self._process_unmatched_detections(detections_array, unmatched_detections)
        self._cleanup_last_seen_objects()
        return self._get_active_objects_centroids()
    
    def _update_staff_visits(self, people: List):
        """Update staff visit information for all active objects"""
        for obj in self.active_objects.values():
            obj.update_last_visit_staff(people, self.frame_count)
    
    def _handle_no_detections(self):
        """Handle the case where no detections are present"""
        objects_to_remove = []
        for obj_id, obj in self.active_objects.items():
            result = obj.handle_disappeared(self.frame_count, self.config, self.should_check_this_frame(), self.current_frame)
            if result == 'move_to_last_seen':
                self._move_to_last_seen(obj_id, "disappeared_too_long")
                objects_to_remove.append(obj_id)
            elif result == 'baked':
                self.total_baked += 1
                objects_to_remove.append(obj_id)
        
        # Remove objects that need to be removed
        for obj_id in objects_to_remove:
            if obj_id in self.active_objects:
                del self.active_objects[obj_id]
    
    def _match_detections_to_active_objects(self, detections_array: np.ndarray) -> set:
        """Match detections to active objects using Hungarian algorithm"""
        obj_ids = list(self.active_objects.keys())
        cost_matrix = self._create_cost_matrix(detections_array, obj_ids)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_detections = set()
        matched_objects = set()
        objects_to_remove = []
        
        for row, col in zip(row_indices, col_indices):
            obj_id = obj_ids[row]
            cost = cost_matrix[row, col]
            if cost < self.config.iou_threshold:
                self.active_objects[obj_id].update_with_detection(
                    detections_array[col], self.current_frame, self.config, self.frame_count)
                matched_detections.add(col)
                matched_objects.add(obj_id)
        
        # Handle unmatched objects
        unmatched_objects = set(obj_ids) - matched_objects
        for obj_id in unmatched_objects:
            obj = self.active_objects[obj_id]
            result = obj.handle_disappeared(self.frame_count, self.config, self.should_check_this_frame, self.current_frame)
            if result == 'move_to_last_seen':
                self._move_to_last_seen(obj_id, "disappeared_too_long")
                objects_to_remove.append(obj_id)
            elif result == 'baked':
                self.total_baked += 1
                objects_to_remove.append(obj_id)
        
        # Remove objects that need to be removed
        for obj_id in objects_to_remove:
            if obj_id in self.active_objects:
                del self.active_objects[obj_id]
        
        return set(range(len(detections_array))) - matched_detections
    
    def _create_cost_matrix(self, detections_array: np.ndarray, obj_ids: List[int]) -> np.ndarray:
        """Create a cost matrix for matching active objects to detections"""
        cost_matrix = np.zeros((len(obj_ids), len(detections_array)))
        for i, obj_id in enumerate(obj_ids):
            obj = self.active_objects[obj_id]
            for j, detection in enumerate(detections_array):
                detection_centroid = np.array([(detection[0] + detection[2]) / 2, 
                                             (detection[1] + detection[3]) / 2])
                detection_pixels = None
                if self.current_frame is not None:
                    # Ensure we only take first 4 values for rect
                    detection_rect = tuple(int(val) for val in detection[:4])
                    # Create a temporary TrackedObject with proper parameters
                    temp_obj = TrackedObject(
                        object_id=0, 
                        centroid=detection_centroid, 
                        rect=detection_rect
                    )
                    detection_pixels = temp_obj.extract_and_resize_pixels(self.current_frame, self.config)
                
                cost_matrix[i, j] = obj.calculate_combined_distance(
                    detection, detection_centroid, detection_pixels, self.config)
        return cost_matrix
    
    def _process_unmatched_detections(self, detections_array: np.ndarray, unmatched_detections: set):
        """Process unmatched detections to re-identify or create new objects"""
        for det_idx in unmatched_detections:
            detection = detections_array[det_idx]
            if not self.last_seen_objects:
                self._create_new_object(detection)
                continue
            
            detection_centroid = np.array([(detection[0] + detection[2]) / 2, 
                                         (detection[1] + detection[3]) / 2])
            
            best_match_id, best_match_score = self._find_best_last_seen_match(detection, detection_centroid)
            if best_match_id is not None:
                self._reactivate_last_seen_object(best_match_id, detection)
            else:
                self._create_new_object(detection)
    
    def _find_best_last_seen_match(self, detection: np.ndarray, detection_centroid: np.ndarray) -> Tuple[Optional[int], float]:
        """Find the best matching object from last_seen based on distance"""
        best_match_id = None
        best_match_score = float('inf')
        
        detection_pixels = None
        if self.current_frame is not None:
            # Ensure we only take first 4 values for rect
            detection_rect = tuple(int(val) for val in detection[:4])
            # Create a temporary TrackedObject with proper parameters
            temp_obj = TrackedObject(
                object_id=0, 
                centroid=detection_centroid, 
                rect=detection_rect
            )
            detection_pixels = temp_obj.extract_and_resize_pixels(self.current_frame, self.config)
        
        for obj_id, last_seen_obj in self.last_seen_objects.items():
            score = last_seen_obj.calculate_combined_distance(
                detection, detection_centroid, detection_pixels, self.config)
            if score < best_match_score and score < self.config.iou_threshold:
                best_match_score = score
                best_match_id = obj_id
        return best_match_id, best_match_score
    
    def _reactivate_last_seen_object(self, obj_id: int, detection: np.ndarray):
        """Reactivate an object from last_seen"""
        obj = self.last_seen_objects[obj_id]
        obj.update_with_detection(detection, self.current_frame, self.config, self.frame_count)
        self.active_objects[obj_id] = obj
        del self.last_seen_objects[obj_id]
        logger.info("Reactivated object %d", obj_id)
        print(f"🔄 Reactivated object {obj_id}")
    
    def _create_new_object(self, bbox: np.ndarray):
        """Create a new tracked object"""
        obj = TrackedObject.create_new(
            self.next_object_id, bbox, self.current_frame, self.config, self.frame_count)
        self.active_objects[self.next_object_id] = obj
        self.next_object_id += 1
    
    def _move_to_last_seen(self, obj_id: int, reason: str = "disappeared"):
        """Move object from active to last_seen with logging"""
        if obj_id in self.active_objects:
            obj = self.active_objects[obj_id]
            self.last_seen_objects[obj_id] = obj
            
            # Count as sale
            self.total_sales += 1
            logger.info("Object %d moved to last_seen and marked as SOLD - Reason: %s", obj_id, reason)
            print(f"💰 Object {obj_id} SOLD - moved to last_seen (Reason: {reason})")
    
    def _cleanup_last_seen_objects(self):
        """Remove long-term disappeared objects from last_seen"""
        objects_to_cleanup = [
            obj_id for obj_id, obj in self.last_seen_objects.items()
            if obj.disappeared_count > self.config.max_disappeared * 2
        ]
        for obj_id in objects_to_cleanup:
            del self.last_seen_objects[obj_id]
            logger.info("Object %d removed from last_seen - Reason: long_term_cleanup", obj_id)
    
    def _get_active_objects_centroids(self) -> Dict[int, np.ndarray]:
        """Return the centroids of active objects"""
        return {obj_id: obj.centroid for obj_id, obj in self.active_objects.items()}
    
    # Public interface methods
    def get_total_sales(self) -> int:
        """Get total number of sales"""
        return self.total_sales
    
    def get_total_baked(self) -> int:
        """Get total number of baked items"""
        return self.total_baked
    
    def get_object_info(self) -> Dict[str, Any]:
        """Return detailed information about tracked objects"""
        return {
            'active_objects': len(self.active_objects),
            'last_seen_objects': len(self.last_seen_objects),
            'total_sales': self.total_sales,
            'total_baked': self.total_baked,
            'frame_count': self.frame_count,
            'active_object_details': {
                obj_id: obj.get_status_dict()
                for obj_id, obj in self.active_objects.items()
            }
        }
    
    def get_full_objects(self) -> Dict[int, Tuple[int, int, int, int]]:
        """Get bounding rectangles of active objects"""
        return {obj_id: obj.rect for obj_id, obj in self.active_objects.items()}
    
    def get_last_seen_objects(self) -> Dict[int, TrackedObject]:
        """Get dictionary of last seen objects"""
        return self.last_seen_objects.copy()
    
    def print_status(self):
        """Print current tracking status"""
        print(f"\n=== Tracking Status (Frame {self.frame_count}) ===")
        print(f"Active Objects: {len(self.active_objects)}")
        print(f"Last Seen Objects: {len(self.last_seen_objects)}")
        print(f"Total Sales: {self.total_sales}")
        print(f"Total Baked: {self.total_baked}")
        
        if self.active_objects:
            print("\nActive Objects Details:")
            for obj_id, obj in self.active_objects.items():
                status = "NEW" if obj.is_new_object else "ESTABLISHED"
                print(f"  ID {obj_id}: disappeared={obj.disappeared_count}, pos={obj.rect}, "
                      f"frames={obj.frames_tracked}, status={status}")
        
        if self.last_seen_objects:
            print("\nLast Seen Objects:")
            for obj_id, obj in self.last_seen_objects.items():
                print(f"  ID {obj_id}: disappeared={obj.disappeared_count}, "
                      f"frames_tracked={obj.frames_tracked}")
        print("=" * 50)