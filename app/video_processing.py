import cv2
import numpy as np
from ultralytics import YOLO
from app.helper import SimpleTracker, TrackerConfig
import os

# Global model variable
model = None

def load_model():
    """Load the YOLO model for staff detection"""
    global model
    if model is None:
        try:
            # Try to load a custom model first, fallback to YOLOv8n
            model_path = "best.pt"  # Your custom trained model
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"Loaded custom model: {model_path}")
            else:
                model = YOLO('yolov8n.pt')
                print("Loaded default YOLOv8n model")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = YOLO('yolov8n.pt')
    return model

def detect_staff(model, frame, original_width=None, original_height=None):
    """Detect staff members in the frame using YOLO model"""
    if model is None:
        model = load_model()
    
    try:
        results = model(frame, verbose=False)
        boxes = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence and class (assuming person is class 0 or specific class)
                    if confidence > 0.5:  # Adjust threshold as needed
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return boxes if boxes else None
        
    except Exception as e:
        print(f"Error in staff detection: {e}")
        return None

def enhance_low_light(frame):
    """Enhance low light conditions in the frame"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except Exception as e:
        print(f"Error in low light enhancement: {e}")
        return frame

def check_staff_in_oven_area(staff_bbox, oven_area):
    """Check if a staff member is in the oven area"""
    if staff_bbox is None or oven_area is None:
        return False
    
    # Extract coordinates
    staff_x1, staff_y1, staff_x2, staff_y2 = staff_bbox
    oven_x1, oven_y1, oven_x2, oven_y2 = oven_area
    
    # Check if staff bounding box overlaps with oven area
    # Calculate intersection
    inter_x1 = max(staff_x1, oven_x1)
    inter_y1 = max(staff_y1, oven_y1)
    inter_x2 = min(staff_x2, oven_x2)
    inter_y2 = min(staff_y2, oven_y2)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        # Calculate intersection area
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        staff_area = (staff_x2 - staff_x1) * (staff_y2 - staff_y1)
        
        # If more than 50% of staff is in oven area, consider them "in the oven"
        overlap_ratio = intersection_area / staff_area if staff_area > 0 else 0
        return overlap_ratio > 0.5
    
    return False

def draw_oven_tracking_overlay(frame, tracked_staff, oven_area, staff_visits):
    """Draw oven area and staff tracking overlay"""
    # Draw oven area
    if oven_area:
        oven_x1, oven_y1, oven_x2, oven_y2 = oven_area
        cv2.rectangle(frame, (oven_x1, oven_y1), (oven_x2, oven_y2), (0, 0, 255), 3)  # Red for oven area
        cv2.putText(frame, 'OVEN AREA', (oven_x1, oven_y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw tracked staff
    for staff_id, bbox in tracked_staff.items():
        x1, y1, x2, y2 = bbox
        
        # Check if staff is in oven area
        is_in_oven = check_staff_in_oven_area(bbox, oven_area)
        
        if is_in_oven:
            # Draw staff in oven with special color (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Staff {staff_id} - IN OVEN', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Draw staff normally (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Staff {staff_id}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Add statistics
    cv2.putText(frame, f'Staff Visits to Oven: {staff_visits}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def writeFrameToVideo(frame, tracker, out, oven_area, staff_visits, scale_factor=1.0):
    """Write frame to video with oven tracking information"""
    try:
        # Get tracked staff
        tracked_staff = tracker.get_full_objects()
        
        # Draw overlay
        frame = draw_oven_tracking_overlay(frame, tracked_staff, oven_area, staff_visits)
        
        out.write(frame)
    except Exception as e:
        print(f"Error writing frame: {e}")

def optimize_memory():
    """Optimize memory usage"""
    import gc
    gc.collect()

def process_video(input_path, output_path, oven_area=None):
    """Process video with staff detection and oven area tracking"""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return 0

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    if oven_area:
        print(f"Oven area: {oven_area}")

    # Reduce frame size if too large (memory optimization)
    if width > 1280 or height > 720:
        scale_factor = min(1280/width, 720/height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        print(f"Resizing video from {width}x{height} to {new_width}x{new_height}")
    else:
        new_width, new_height = width, height
        scale_factor = 1.0
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Load model
    model = load_model()
    
    # Initialize the tracker
    config = TrackerConfig()
    staff_tracker = SimpleTracker(config)
    
    processed_frames = 0
    staff_visits = 0
    staff_in_oven_frames = set()  # Track unique staff visits
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            # Resize frame if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            # Enhance low light if needed
            frame = enhance_low_light(frame)

            # Perform staff detection
            staff_boxes = detect_staff(model, frame, width, height)
            
            # Update tracker
            objects = staff_tracker.update(staff_boxes, frame)
            
            # Check for staff in oven area
            if oven_area and staff_boxes:
                for staff_bbox in staff_boxes:
                    if check_staff_in_oven_area(staff_bbox, oven_area):
                        # Get staff ID from tracker
                        for staff_id, tracked_bbox in staff_tracker.get_full_objects().items():
                            if (abs(staff_bbox[0] - tracked_bbox[0]) < 10 and 
                                abs(staff_bbox[1] - tracked_bbox[1]) < 10):
                                staff_in_oven_frames.add(staff_id)
                                break
            
            # Write frame to video
            writeFrameToVideo(frame, staff_tracker, out, oven_area, len(staff_in_oven_frames), scale_factor)
            
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()

    finally:
        # Always release resources
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Oven tracking completed!")
        print(f"Output saved as: {output_path}")
        print(f"Total frames processed: {processed_frames}")
        print(f"Unique staff visits to oven: {len(staff_in_oven_frames)}")
        
        # Final memory cleanup
        optimize_memory()
        
        return len(staff_in_oven_frames)
