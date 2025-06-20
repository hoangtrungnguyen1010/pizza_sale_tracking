# Project: Pixxa Count

## Overview
Pixxa Count is a video processing tool designed to track and analyze pizza sales using computer vision. The application allows users to interact with video frames, select objects of interest, and process videos to extract useful insights.

## Features
- Drag-and-drop functionality to select bounding boxes on videos.
- Real-time video display with interactive overlays.
- Backend processing to analyze selected regions in videos.
- Outputs processed videos for review.

## Project Structure
```
pixxa_count/
├── app/
│   ├── static/                  # Frontend assets
│   │   ├── css/                 # Styling
│   │   ├── js/                  # JavaScript for interactivity
│   └── templates/               # HTML templates
│       └── index.html           # Main UI
├── data/                        # Uploaded and processed videos
│   ├── sample_video.mp4         # Example video
│   ├── output/                  # Directory for processed videos
├── app/video_processing.py      # Python logic for video detection
├── app/server.py                # Flask backend
├── Dockerfile                   # Docker setup
├── docker-compose.yml           # Compose services
├── requirements.txt             # Python dependencies
└── README.md                    # Instructions
```

##![Diagram](images/diagram.png)

## Requirements
- Python 3.10+
- Docker & Docker Compose
- Flask
- OpenCV

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pixxa_count.git
   cd pixxa_count
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
1. Upload a video file to the `data/` directory (e.g., `sample_video.mp4`).
2. Open the web interface at `http://localhost:5000`.
3. Play the video and drag a rectangle over the region of interest.
4. Click "Start Processing" to analyze the selected region.
5. View the processed video in the "Processed Video" section.

## API Endpoints

### `/start-processing`
- **Method**: POST
- **Description**: Processes the video based on user-selected bounding box coordinates.
- **Payload**:
  ```json
  {
      "startX": 100,
      "startY": 50,
      "endX": 200,
      "endY": 150
  }
  ```
- **Response**:
  ```json
  {
      "status": "success",
      "outputVideoPath": "/static/output/processed_video.mp4"
  }
  ```

### `/static/<filename>`
- **Method**: GET
- **Description**: Serves static files such as videos and processed outputs.

## Notes
- Ensure Docker is running before starting the application.
- The processed video will be saved in the `data/output/` directory.

## Troubleshooting
- **TclError: Couldn't connect to display**:
  - Ensure you are running the application in a headless environment using Xvfb.
  - Update the Dockerfile to include:
    ```bash
    CMD xvfb-run python main.py
    ```

- **Bounding Box Not Working**:
  - Check the browser console for errors.
  - Ensure the JavaScript logic in `index.html` is correct.

## License
MIT License
