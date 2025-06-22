"""
Video processing module with clean, object-oriented design.
Handles video processing, object detection, and visualization.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from .core.objects import Pizza, Person
from .core.trackers import PizzaTracker, PersonTracker
from .core.base import BoundingBox
from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages YOLO model loading and caching."""
    
    def __init__(self):
        self._model: Optional[YOLO] = None
    
    def load_model(self) -> YOLO:
        """Load and cache the YOLO model for object detection."""
        if self._model is None:
            try:
                # Try to load custom model first, fallback to default
                model_path = config.detection.model_path
                if os.path.exists(model_path):
                    self._model = YOLO(model_path)
                    logger.info(f"Loaded custom model: {model_path}")
                else:
                    self._model = YOLO('yolov8n.pt')
                    logger.info("Loaded default YOLOv8n model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._model = YOLO('yolov8n.pt')
        
        return self._model
    
    @property
    def model(self) -> YOLO:
        """Get the loaded model."""
        return self.load_model()


class ObjectDetector:
    """Handles object detection using YOLO model."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def detect_objects(self, frame: np.ndarray, 
                      confidence_threshold: Optional[float] = None,
                      target_classes: Optional[List[int]] = None) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect objects in the frame using YOLO model.
        
        Args:
            frame: Input frame
            confidence_threshold: Minimum confidence for detection
            target_classes: List of class IDs to detect (None for all)
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        if confidence_threshold is None:
            confidence_threshold = config.detection.min_person_confidence
        
        try:
            model = self.model_manager.model
            results = model(frame, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Apply filters
                        if (confidence >= confidence_threshold and 
                            (target_classes is None or class_id in target_classes)):
                            detections.append((
                                int(x1), int(y1), int(x2), int(y2), confidence
                            ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect people in the frame."""
        return self.detect_objects(frame, 
                                 confidence_threshold=config.detection.min_person_confidence,
                                 target_classes=[0])  # Class 0 is typically person
    
    def detect_pizzas(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect pizzas in the frame."""
        return self.detect_objects(frame, 
                                 confidence_threshold=config.detection.min_pizza_confidence,
                                 target_classes=[config.detection.pizza_class_id])


class ImageProcessor:
    """Handles image processing and enhancement."""
    
    @staticmethod
    def enhance_low_light(frame: np.ndarray) -> np.ndarray:
        """
        Enhance low light conditions in the frame using CLAHE.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=config.image_processing.clahe_clip_limit,
                tileGridSize=config.image_processing.clahe_tile_grid_size
            )
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in low light enhancement: {e}")
            return frame
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Resize frame to target dimensions."""
        return cv2.resize(frame, (target_width, target_height))
    
    @staticmethod
    def calculate_scale_factor(original_width: int, original_height: int, 
                             max_width: int = 1280, max_height: int = 720) -> float:
        """Calculate scale factor for resizing."""
        if original_width <= max_width and original_height <= max_height:
            return 1.0
        
        scale_factor = min(max_width / original_width, max_height / original_height)
        return scale_factor


class VideoVisualizer:
    """Handles video visualization and overlay drawing."""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.display.font_scale
        self.font_thickness = config.display.font_thickness
    
    def draw_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         color: Tuple[int, int, int], label: str = "", 
                         thickness: int = 2) -> np.ndarray:
        """Draw a bounding box with optional label."""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            # Calculate text position
            text_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 10
            
            # Draw text background
            cv2.rectangle(frame, (text_x, text_y - text_size[1]), 
                         (text_x + text_size[0], text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (text_x, text_y), self.font, 
                       self.font_scale, (255, 255, 255), self.font_thickness)
        
        return frame
    
    def draw_oven_area(self, frame: np.ndarray, oven_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw the oven area on the frame."""
        return self.draw_bounding_box(frame, oven_bbox, (0, 0, 255), "OVEN AREA", 3)
    
    def draw_person(self, frame: np.ndarray, person: Person, 
                   is_in_oven: bool = False) -> np.ndarray:
        """Draw a person on the frame."""
        color = (0, 255, 0) if is_in_oven else (255, 0, 0)  # Green if in oven, blue otherwise
        label = f"Staff {person.object_id}" + (" - IN OVEN" if is_in_oven else "")
        
        return self.draw_bounding_box(frame, person.position, color, label)
    
    def draw_pizza(self, frame: np.ndarray, pizza: Pizza) -> np.ndarray:
        """Draw a pizza on the frame."""
        return self.draw_bounding_box(frame, pizza.position, config.display.pizza_color, 
                                    f"Pizza {pizza.object_id}")
    
    def draw_statistics(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw statistics on the frame."""
        y_offset = config.display.text_margin
        
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), self.font, 
                       self.font_scale, (255, 255, 255), self.font_thickness)
            y_offset += config.display.text_spacing
        
        return frame


class VideoProcessor:
    """Main video processing class that orchestrates the entire pipeline."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.detector = ObjectDetector(self.model_manager)
        self.image_processor = ImageProcessor()
        self.visualizer = VideoVisualizer()
        
        # Initialize trackers
        self.person_tracker = PersonTracker()
        self.pizza_tracker = PizzaTracker()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'people_detected': 0,
            'pizzas_detected': 0,
            'staff_oven_visits': 0,
            'pizzas_baked': 0,
            'pizzas_sold': 0
        }
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Process a single frame through the detection and tracking pipeline.
        
        Args:
            frame: Input frame
            frame_number: Current frame number
            
        Returns:
            Processed frame with overlays
        """
        try:
            # Enhance image if needed
            if config.video.enable_enhancement:
                frame = self.image_processor.enhance_low_light(frame)
            
            # Detect objects
            people_detections = self.detector.detect_people(frame)
            pizza_detections = self.detector.detect_pizzas(frame)
            
            # Update trackers
            self.person_tracker.update(people_detections, frame)
            
            # Get active people for pizza tracker
            active_people_dict = self.person_tracker.get_active_objects()
            active_people_list = [obj for obj in active_people_dict.values() if isinstance(obj, Person)]
            
            self.pizza_tracker.update(pizza_detections, frame, people=active_people_list)
            
            # Get active objects (actual objects, not dictionaries)
            active_people = self.person_tracker.get_active_objects()
            active_pizzas = self.pizza_tracker.get_active_objects()
            
            # Cast to proper types for drawing
            people_dict = {k: v for k, v in active_people.items() if isinstance(v, Person)}
            pizzas_dict = {k: v for k, v in active_pizzas.items() if isinstance(v, Pizza)}
            
            # Update statistics
            self.stats['total_frames'] = frame_number
            self.stats['people_detected'] = len(people_dict)
            self.stats['pizzas_detected'] = len(pizzas_dict)
            
            # Draw overlays
            frame = self._draw_overlays(frame, people_dict, pizzas_dict)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return frame
    
    def _draw_overlays(self, frame: np.ndarray, 
                      active_people: Dict[int, Person], 
                      active_pizzas: Dict[int, Pizza]) -> np.ndarray:
        """Draw all overlays on the frame."""
        # Draw oven area
        oven_bbox = config.person.oven_bbox
        frame = self.visualizer.draw_oven_area(frame, oven_bbox)
        
        # Draw people
        for person in active_people.values():
            is_in_oven = person.check_if_went_to_oven()
            frame = self.visualizer.draw_person(frame, person, is_in_oven)
        
        # Draw pizzas
        for pizza in active_pizzas.values():
            frame = self.visualizer.draw_pizza(frame, pizza)
        
        # Draw statistics
        stats = {
            'Frame': self.stats['total_frames'],
            'People': self.stats['people_detected'],
            'Pizzas': self.stats['pizzas_detected'],
            'Oven Visits': self.stats['staff_oven_visits'],
            'Baked': self.stats['pizzas_baked'],
            'Sold': self.stats['pizzas_sold']
        }
        frame = self.visualizer.draw_statistics(frame, stats)
        
        return frame
    
    def process_video(self, input_path: str, output_path: str, oven_area: Optional[Tuple[int, int, int, int]] = None, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a video file through the complete pipeline.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            oven_area: Optional user-defined oven area
            max_frames: Optional limit on number of frames to process (for testing)
            
        Returns:
            Dictionary with processing statistics
        """
        # Update oven area in config if provided
        if oven_area is not None:
            config.person.oven_bbox = oven_area
            logger.info(f"Updated oven area to: {oven_area}")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Limit frames if specified
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)
                logger.info(f"Limiting processing to {max_frames} frames for testing")
            
            logger.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Calculate scale factor
            scale_factor = self.image_processor.calculate_scale_factor(width, height)
            if scale_factor != 1.0:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                logger.info(f"Resizing video to {new_width}x{new_height}")
            else:
                new_width, new_height = width, height
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            
            frame_count = 0
            
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize frame if needed
                if scale_factor != 1.0:
                    frame = self.image_processor.resize_frame(frame, new_width, new_height)
                
                # Process frame
                processed_frame = self.process_frame(frame, frame_count)
                
                # Write frame
                out.write(processed_frame)
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Get final statistics
            final_stats = self._get_final_statistics()
            
            logger.info("Video processing completed successfully!")
            return final_stats
            
        finally:
            cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
    
    def _get_final_statistics(self) -> Dict[str, Any]:
        """Get final processing statistics."""
        person_stats = self.person_tracker.get_statistics()
        pizza_stats = self.pizza_tracker.get_statistics()
        
        return {
            'total_frames': self.stats['total_frames'],
            'people_detected': self.stats['people_detected'],
            'pizzas_detected': self.stats['pizzas_detected'],
            'pizzas_baked': pizza_stats.get('total_baked', 0),
            'pizzas_sold': pizza_stats.get('total_sales', 0),
            'person_tracker_stats': person_stats,
            'pizza_tracker_stats': pizza_stats
        }


# Backward compatibility functions
def load_model():
    """Backward compatibility function for model loading."""
    model_manager = ModelManager()
    return model_manager.load_model()


def detect_staff(model, frame, original_width=None, original_height=None):
    """Backward compatibility function for staff detection."""
    model_manager = ModelManager()
    model_manager._model = model
    detector = ObjectDetector(model_manager)
    return detector.detect_people(frame)


def enhance_low_light(frame):
    """Backward compatibility function for image enhancement."""
    return ImageProcessor.enhance_low_light(frame)


def process_video(input_path, output_path, oven_area=None, max_frames=None):
    """Backward compatibility function for video processing."""
    processor = VideoProcessor()
    
    # Update oven area in config if provided
    if oven_area is not None:
        # Update the config with the user-defined oven area
        config.person.oven_bbox = oven_area
        logger.info(f"Updated oven area to: {oven_area}")
    
    return processor.process_video(input_path, output_path, oven_area, max_frames)
