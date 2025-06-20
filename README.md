# Project: Pixxa Count

## Overview
Pixxa Count is an advanced computer vision application designed to count pizza sales by tracking customer interactions and transactions. The system uses Region of Interest (ROI) detection, enhanced people tracking, and sophisticated video processing to provide insights into sales patterns and customer behavior at pizza establishments.

## Features

### 🔍 **Region of Interest (ROI) Detection**
- **Interactive ROI Selection**: Users can draw a rectangle around the sales/counter area to define the Region of Interest
- **Precise Area Definition**: The system tracks customer interactions specifically within the defined sales area
- **Real-time Visualization**: The ROI is highlighted in red-orange color with "SALES AREA" label
- **Customizable Boundaries**: Adjustable ROI size and position for different store layouts

### 👥 **Enhanced People Detection**
- **YOLO-based Detection**: Uses Ultralytics YOLO models for accurate customer detection
- **ROI Cropping**: Automatically crops the region that includes people for better detection accuracy
- **Multi-person Tracking**: Simultaneously tracks multiple customers
- **Confidence Filtering**: Filters detections based on confidence thresholds for reliability

### 🌟 **Low Light Enhancement**
- **CLAHE Algorithm**: Implements Contrast Limited Adaptive Histogram Equalization
- **LAB Color Space Processing**: Converts to LAB color space for better light enhancement
- **Automatic Enhancement**: Applies enhancement to all frames for consistent visibility
- **Performance Optimized**: Efficient processing without significant performance impact

### 🔄 **Occlusion Handling**
- **Object Persistence**: Maintains object identity even when temporarily occluded
- **Disappearance Tracking**: Tracks how long objects remain undetected
- **Reappearance Detection**: Automatically re-identifies objects when they reappear
- **Historical Data**: Uses object history for better tracking during occlusions
- **Pixel Similarity Matching**: Compares object appearances for re-identification

### 📊 **Sales Analytics**
- **Customer Visit Counting**: Tracks unique customer visits to the sales area
- **Transaction Analysis**: Measures time spent in the sales area (potential transactions)
- **Sales Pattern Insights**: Provides data on peak hours and customer flow
- **Real-time Statistics**: Displays live tracking statistics on video overlay

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  ROI Selection  │───▶│ People Detection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Low Light      │◀───│  Video          │◀───│ Object Tracking │
│ Enhancement     │    │ Processing      │    │ & Occlusion     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Analytics &     │◀───│ Sales Area      │◀───│ Visit Detection │
│ Reporting       │    │ Interaction     │    │ & Counting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

![System Architecture](image/diagram.png)

## Technical Implementation

### ROI Processing Pipeline
1. **User Input**: Interactive rectangle drawing on video
2. **Area Definition**: Coordinates stored for processing
3. **Crop Generation**: Automatic cropping of ROI for detection
4. **Detection Focus**: People detection focused within ROI
5. **Interaction Analysis**: Customer-sales area interaction tracking

### People Detection Enhancement
```python
def detect_customers(model, frame, roi_area=None):
    """
    Enhanced customer detection with ROI cropping
    - Crops frame to ROI for better detection
    - Applies confidence filtering
    - Returns bounding boxes with customer IDs
    """
```

### Low Light Enhancement
```python
def enhance_low_light(frame):
    """
    CLAHE-based low light enhancement
    - Converts to LAB color space
    - Applies CLAHE to L channel
    - Maintains color accuracy
    """
```

### Occlusion Handling
```python
def handle_occlusion(tracked_object, current_frame):
    """
    Advanced occlusion handling
    - Tracks disappearance duration
    - Uses pixel similarity for re-identification
    - Maintains object history
    """
```

## Project Structure
```
pixxa_count/
├── app/
│   ├── static/                  # Frontend assets
│   │   ├── css/                 # Styling
│   │   ├── js/                  # JavaScript for interactivity
│   └── templates/               # HTML templates
│       └── index.html           # Main UI with ROI selection
├── app/helper/                  # Core tracking logic
│   ├── TrackerConfig.py         # Configuration parameters
│   ├── trackedObject.py         # Object tracking data classes
│   ├── pizzaTracker.py          # Main tracking algorithms
│   └── __init__.py              # Package initialization
├── data/                        # Uploaded and processed videos
│   ├── uploads/                 # User uploaded videos
│   ├── output/                  # Processed videos with tracking
│   └── sample_video.mp4         # Example video
├── image/                       # Documentation images
│   └── diagram.png              # System architecture diagram
├── app/video_processing.py      # Video processing with ROI
├── app/server.py                # Flask backend with ROI handling
├── Dockerfile                   # Docker setup
├── docker-compose.yml           # Compose services
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Requirements
- Python 3.10+
- Docker & Docker Compose
- Flask
- OpenCV
- Ultralytics YOLO
- NumPy
- SciPy
- Scikit-image

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hoangtrungnguyen1010/pizza_sale_tracking.git
   cd pizza_sale_tracking
   ```

2. Build the Docker container:
   ```bash
   docker-compose build
   ```

3. Start the application:
   ```bash
   docker-compose up
   ```

4. Access the application in your browser:
   ```
   http://localhost:5000
   ```

## Usage

### 1. **Upload Video**
   - Select a video file showing the sales/counter area
   - Supported format: MP4
   - Video should clearly show customer interactions at the counter

### 2. **Define ROI (Sales Area)**
   - Draw a rectangle around the sales/counter area by clicking and dragging
   - The ROI defines the area where customer visits will be tracked
   - Ensure the rectangle covers the entire sales interaction area

### 3. **Start Tracking**
   - Click "Start Sales Tracking" to begin analysis
   - The system will process the video with ROI-based detection
   - Real-time tracking overlay shows customer movements

### 4. **Review Results**
   - View the processed video with tracking overlay
   - Red rectangle: Sales area (ROI)
   - Blue rectangles: Customers
   - Green rectangles: Customers currently in sales area
   - Statistics: Total customer visits to sales area

## API Endpoints

### `/upload`
- **Method**: POST
- **Description**: Upload video file for processing
- **Payload**: Multipart form data with video file
- **Response**: Success status and filename

### `/process`
- **Method**: POST
- **Description**: Process video with ROI-based customer tracking
- **Payload**:
  ```json
  {
      "startX": 100,
      "startY": 50,
      "width": 200,
      "height": 150,
      "filename": "video.mp4",
      "salesArea": true
  }
  ```
- **Response**:
  ```json
  {
      "success": true,
      "output_video": "/output/sales_tracking_video.mp4",
      "message": "Sales area tracking completed! Found 15 customer visits to the sales area.",
      "customer_visits": 15,
      "sales_area": [100, 50, 300, 200]
  }
  ```

### `/output/<filename>`
- **Method**: GET
- **Description**: Download processed videos

## Key Features Explained

### ROI-Based Detection
The system uses Region of Interest (ROI) to focus detection efforts on the most important area - the sales counter. This improves:
- **Detection Accuracy**: Better people detection within the ROI
- **Processing Speed**: Reduced computational load
- **Relevance**: Focus on meaningful sales interactions

### Low Light Enhancement
Store environments often have challenging lighting conditions. The system addresses this with:
- **CLAHE Algorithm**: Adaptive histogram equalization
- **Color Space Processing**: LAB color space for better enhancement
- **Automatic Application**: Applied to all frames consistently

### Occlusion Handling
Customers often move behind equipment or each other. The system handles this with:
- **Object Persistence**: Maintains identity during temporary occlusion
- **Pixel Similarity**: Re-identification using appearance matching
- **Historical Tracking**: Uses movement patterns for prediction

## Troubleshooting

### **Detection Issues**
- Ensure good lighting in the video
- Make sure the ROI covers the entire sales area
- Check that customers are clearly visible

### **Performance Issues**
- Reduce video resolution if processing is slow
- Ensure sufficient system resources
- Check Docker container resource limits

### **Tracking Accuracy**
- Adjust ROI size for better coverage
- Ensure consistent lighting conditions
- Use higher quality video input

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
