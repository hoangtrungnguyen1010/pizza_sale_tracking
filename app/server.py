from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from app.video_processing import process_video

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
OUTPUT_FOLDER = 'data/output'
ALLOWED_EXTENSIONS = {'mp4'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        start_x = int(data['startX'])
        start_y = int(data['startY'])
        width = int(data['width'])
        height = int(data['height'])
        filename = data.get('filename', 'sample_video.mp4')  # Default fallback

        # Use uploaded video or fallback to sample
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            # Fallback to sample video if uploaded file doesn't exist
            input_path = os.path.join('data', 'sample_video.mp4')
            if not os.path.exists(input_path):
                return jsonify({'error': 'No video file found'}), 404

        output_filename = f'processed_{filename}'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Call process_video with coordinates
        process_video(input_path, output_path, (start_x, start_y, start_x + width, start_y + height))

        return jsonify({
            'success': True,
            'output_video': f'/output/{output_filename}',
            'message': 'Video processed successfully!'
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/output/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
