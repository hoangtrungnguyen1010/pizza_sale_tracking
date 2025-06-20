import numpy as np
import cv2
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o
from ultralytics import YOLO
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Install required packages (run this cell first if needed)
# !pip install scipy filterpy

# Initialize YOLO11 model
model = YOLO('yolo11n.pt')


PIZZA_CLASS_ID = 53
# Video path
# Configuration
MIN_PIZZA_CONFIDENCE = 0.3


def expand_bbox(x1, y1, x2, y2, frame_width, frame_height, expand_ratio=0.1):
    """
    Expands the bounding box by a given ratio while ensuring it stays within the frame bounds.

    Args:
        x1 (int): Top-left x-coordinate of the bounding box.
        y1 (int): Top-left y-coordinate of the bounding box.
        x2 (int): Bottom-right x-coordinate of the bounding box.
        y2 (int): Bottom-right y-coordinate of the bounding box.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
        expand_ratio (float): Ratio by which to expand the bounding box on each side.

    Returns:
        Tuple[int, int, int, int]: New bounding box coordinates (x1, y1, x2, y2).
    """
    # Calculate width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Calculate the amount to expand on each side
    expand_x = bbox_width * expand_ratio
    expand_y = bbox_height * expand_ratio

    # Expand the bounding box
    new_x1 = max(0, int(x1 - expand_x))
    new_y1 = max(0, int(y1 - expand_y))
    new_x2 = min(frame_width, int(x2 + expand_x))
    new_y2 = min(frame_height, int(y2 + expand_y))

    return new_x1, new_y1, new_x2, new_y2


def detect_pizzas(model, frame, tracker, width, height):
    """
    Detect pizzas in a frame with a two-stage process:
    1. Detect people and pizzas in the full frame.
    2. For stationary people, crop their surrounding region and detect pizzas.
    """
    # STAGE 1: Detect people in the full frame
    human_results = model(frame, verbose=False)  # Changed `predict_frame` to `frame`
    
    detections = []
    pizza_detections = []

    # Process human and pizza detections
    for result in human_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Process people and direct pizza detections
                if class_name == 'person' and confidence > 0.3:
                    detections.append((x1, y1, x2, y2, confidence))
                elif class_name == 'pizza' and confidence > 0.3:
                    pizza_detections.append((x1, y1, x2, y2, confidence))
    
    # Update tracker with new detections
    people = tracker.update(detections)
    
    # STAGE 2: Detect pizzas near stationary people
    pizza_through_stationary_people = []
    
    for person in people:
        if not person.is_moving:  # Only process stationary people
            hx1, hy1, hx2, hy2 = person.get_bbox()
            
            # Expand bounding box and ensure it remains within frame bounds
            crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
                hx1, hy1, hx2, hy2, width, height
            )
            
            # Crop region around stationary person
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped_frame.size > 0:  # Ensure valid crop
                # Enhance cropped region for detection
                enhanced_crop = enhance_low_light(cropped_frame)
                
                # Detect pizzas in the cropped region
                crop_results = model(enhanced_crop, verbose=False)
                for result in crop_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get detection info
                            cx1, cy1, cx2, cy2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            if class_name == 'pizza' and confidence > 0.15:
                                # Map cropped coordinates to full frame
                                full_x1 = crop_x1 + cx1
                                full_y1 = crop_y1 + cy1
                                full_x2 = crop_x1 + cx2
                                full_y2 = crop_y1 + cy2
                                
                                pizza_through_stationary_people.append((
                                    full_x1, full_y1, full_x2, full_y2, confidence
                                ))

    # Combine detections
    total_pizza = pizza_detections + pizza_through_stationary_people
    return np.array(total_pizza) if total_pizza else None

def draw_tracked_pizzas(frame, tracked_objects):
    """Draw tracked pizzas on frame"""
    for track_id, obj in tracked_objects.items():
        x1, y1, x2, y2= obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Draw bounding box in green
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track ID
        cv2.putText(frame, f"Pizza #{track_id}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    return frame

def writeFrameToVideo(frame, tracker, out, scale_factor=1.0):
    """
    Process and annotate a video frame, then write it to the output video.
    
    Args:
        frame (np.array): The current video frame.
        tracker: The tracking object with pizza and other detections.
        out: The video writer object.
        scale_factor (float): Scale factor for adjusting text size.
    """
    # Draw tracked pizzas on the frame
    frame = draw_tracked_pizzas(frame, tracker.get_full_objects())
    
    # Set font scale and thickness based on scale factor
    font_scale = 0.8 if scale_factor < 1.0 else 1.0
    thickness = 2 if scale_factor < 1.0 else 3
    
    # Add semi-transparent background for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (450, 140), (0, 0, 0), -1)  # Black rectangle
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # Blend overlay with frame
    
    # Annotate total unique pizza sales (Bright Green)
    cv2.putText(
        frame, 
        f"Total Unique Pizzas Sales: {tracker.get_total_sales()}", 
        (10, 50), 
        cv2.FONT_HERSHEY_DUPLEX, 
        font_scale, 
        (0, 255, 0), 
        thickness
    )
    
    # Annotate total active objects detected (Bright Green)
    cv2.putText(
        frame, 
        f"Total Active Objects Detected: {len(tracker.active_objects)}", 
        (10, 100), 
        cv2.FONT_HERSHEY_DUPLEX, 
        font_scale, 
        (0, 255, 0), 
        thickness
    )
    
    # Write the annotated frame to the output video
    out.write(frame)

import cv2
import numpy as np
def enhance_low_light(frame):
    """
    Recommended version: LAB CLAHE (preserves color, good enhancement)
    This is likely the best balance for your pizza detection task
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Optional: Add slight brightness boost
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=15)
    
    return enhanced


