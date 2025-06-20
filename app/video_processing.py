import cv2
import numpy as np
from ultralytics import YOLO
from app.helper import SimpleTracker, TrackerConfig
import os

# Global model variable
model = None

def load_model():
    """Load the YOLO model for pizza detection"""
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

def detect_pizzas(model, frame, staff_tracker=None, original_width=None, original_height=None):
    """Detect pizzas in the frame using YOLO model"""
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
                    
                    # Filter by confidence and class (assuming pizza is class 0 or specific class)
                    if confidence > 0.5:  # Adjust threshold as needed
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return boxes if boxes else None
        
    except Exception as e:
        print(f"Error in pizza detection: {e}")
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

def draw_tracked_pizzas(frame, tracked_objects):
    """Draw bounding boxes for tracked pizzas"""
    for obj_id, bbox in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Pizza {obj_id}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def writeFrameToVideo(frame, tracker, out, scale_factor=1.0):
    """Write frame to video with tracking information"""
    try:
        # Add tracking statistics to frame
        total_sales = tracker.get_total_sales()
        cv2.putText(frame, f'Total Pizzas: {total_sales}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    except Exception as e:
        print(f"Error writing frame: {e}")

def optimize_memory():
    """Optimize memory usage"""
    import gc
    gc.collect()

def process_video(input_path, output_path, bounding_box=None):
    """Process video with pizza detection and tracking"""
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
    pizza_tracker = SimpleTracker(config)
    
    processed_frames = 0
    
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

            # Perform object detection
            boxes = detect_pizzas(model, frame, None, width, height)
            
            # Update tracker
            objects = pizza_tracker.update(boxes, frame)
            
            # Draw tracked objects
            frame = draw_tracked_pizzas(frame, pizza_tracker.get_full_objects())
            
            # Write frame to video
            writeFrameToVideo(frame, pizza_tracker, out, scale_factor)
            
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
        
        print(f"Video processing completed!")
        print(f"Output saved as: {output_path}")
        print(f"Total frames processed: {processed_frames}")
        
        total_sales = 0
        if 'pizza_tracker' in locals():
            total_sales = pizza_tracker.get_total_sales()
            print(f"Total unique pizzas tracked: {total_sales}")
        
        # Final memory cleanup
        optimize_memory()
        
        return total_sales
