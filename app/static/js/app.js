// Video and canvas overlay logic for drawing rectangle on uploaded video
window.addEventListener('DOMContentLoaded', function() {
    const uploadBtn = document.getElementById('uploadBtn');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let rect = {};
    let hasRect = false;
    let videoLoaded = false;
    let uploadedFilename = null; // Store the uploaded filename

    // Video error handling print error message
    video.addEventListener('error', function(e) {
        let error = video.error;
        let message = 'Error: Unable to load the video.';
        if (error) {
            switch (error.code) {
                case error.MEDIA_ERR_ABORTED:
                    message += ' You aborted the video playback.';
                    break;
                case error.MEDIA_ERR_NETWORK:
                    message += ' A network error caused the video download to fail.';
                    break;
                case error.MEDIA_ERR_DECODE:
                    message += ' The video playback was aborted due to a corruption problem or because the video used features your browser did not support.';
                    break;
                case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                    message += ' The video could not be loaded, either because the server or network failed or because the format is not supported.';
                    break;
                default:
                    message += ' An unknown error occurred.';
                    break;
            }
            message += ` (Error code: ${error.code})`;
        }
        alert(message);
        console.error('Video error:', error);
    });
    video.addEventListener('error', function() {
        alert('Error: Unable to load the video. Please make sure you selected a valid MP4 file.');
        console.error('Video failed to load.');
    });

    // Sync canvas size with video dimensions
    video.addEventListener('loadedmetadata', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        video.style.width = '100%';
        video.style.height = '100%';
        videoLoaded = true;
        clearRectAndButton();
        drawOverlay();
        console.log('Video metadata loaded, canvas synced.');
    });

    // Always draw overlay (rectangle) on top of video
    function drawOverlay() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (hasRect && rect.width && rect.height) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(rect.startX, rect.startY, rect.width, rect.height);
        }
    }

    // Redraw overlay on every frame (if video is playing)
    video.addEventListener('play', () => {
        function drawFrame() {
            if (!video.paused && !video.ended) {
                drawOverlay();
                requestAnimationFrame(drawFrame);
            }
        }
        drawFrame();
    });

    // Also redraw overlay when video is paused or seeked
    video.addEventListener('pause', drawOverlay);
    video.addEventListener('seeked', drawOverlay);

    // Rectangle drawing logic
    canvas.addEventListener('mousedown', (e) => {
        if (!videoLoaded) {
            console.log('mousedown: video not loaded');
            return;
        }
        isDrawing = true;
        hasRect = false;
        const rectLeft = canvas.getBoundingClientRect();
        rect.startX = (e.clientX - rectLeft.left) * (canvas.width / canvas.offsetWidth);
        rect.startY = (e.clientY - rectLeft.top) * (canvas.height / canvas.offsetHeight);
        rect.width = 0;
        rect.height = 0;
        canvas.classList.add('drawing');
        canvas.style.pointerEvents = 'auto';
        console.log('mousedown:', {
            x: e.clientX,
            y: e.clientY,
            startX: rect.startX,
            startY: rect.startY,
            isDrawing
        });
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) {
            //console.log('mousemove: not drawing');
            return;
        }
        const rectLeft = canvas.getBoundingClientRect();
        const currX = (e.clientX - rectLeft.left) * (canvas.width / canvas.offsetWidth);
        const currY = (e.clientY - rectLeft.top) * (canvas.height / canvas.offsetHeight);
        rect.width = currX - rect.startX;
        rect.height = currY - rect.startY;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'rgba(255,0,0,0.7)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6]);
        ctx.strokeRect(rect.startX, rect.startY, rect.width, rect.height);
        ctx.setLineDash([]);
        console.log('mousemove:', {
            x: e.clientX,
            y: e.clientY,
            currX,
            currY,
            width: rect.width,
            height: rect.height,
            isDrawing
        });
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!isDrawing) {
            console.log('mouseup: not drawing');
            return;
        }
        isDrawing = false;
        hasRect = Math.abs(rect.width) > 10 && Math.abs(rect.height) > 10;
        canvas.classList.remove('drawing');
        canvas.style.pointerEvents = 'none';
        drawOverlay();
        clearRectAndButton();
        console.log('mouseup:', {
            x: e.clientX,
            y: e.clientY,
            hasRect,
            rect
        });
    });

    function clearRectAndButton() {
        // Enable process button only if video loaded and rectangle drawn
        const btn = document.getElementById('processBtn');
        if (videoLoaded && hasRect) {
            btn.disabled = false;
            btn.setAttribute('title', '');
        } else {
            btn.disabled = true;
            btn.setAttribute('title', 'Draw a rectangle and upload a video first');
        }
    }

    // Handle file upload and video load
    uploadBtn.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;
        // Accept if file.type is video/mp4 or file name ends with .mp4 (case-insensitive)
        const isMp4 = file.type === 'video/mp4' || file.name.toLowerCase().endsWith('.mp4');
        if (!isMp4) {
            alert('Please select a valid MP4 video file.');
            console.error('Invalid file type:', file.type, file.name);
            return;
        }

        // Upload file to server
        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFilename = data.filename;
                console.log('File uploaded successfully:', uploadedFilename);
                
                // Load video in browser
                const videoURL = URL.createObjectURL(file);
                video.src = videoURL;
                video.load();
                hasRect = false;
                rect = {};
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                clearRectAndButton();
                console.log('Video file uploaded and src set.');
            } else {
                alert('Upload failed: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Upload error:', error);
            alert('Upload failed. Please try again.');
        });
    });

    // Handle the processing button
    document.getElementById('processBtn').addEventListener('click', () => {
        if (!hasRect || !videoLoaded) {
            alert('Please draw a rectangle on the video first.');
            return;
        }

        // Show loading overlay with initial message
        const spinnerOverlay = document.getElementById('spinnerOverlay');
        const spinnerText = spinnerOverlay.querySelector('.spinner-text');
        const spinnerSubtext = spinnerOverlay.querySelector('.spinner-subtext');
        
        spinnerText.textContent = 'Processing Video...';
        spinnerSubtext.textContent = 'Drawing rectangle on each frame...';
        spinnerOverlay.style.display = 'flex';
        
        const processData = {
            startX: rect.startX,
            startY: rect.startY,
            width: rect.width,
            height: rect.height,
            filename: uploadedFilename
        };

        // Update message after a short delay to show progress
        setTimeout(() => {
            spinnerSubtext.textContent = 'Applying effects to video frames...';
        }, 2000);

        fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(processData),
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            spinnerText.textContent = 'Processing Complete!';
            spinnerSubtext.textContent = 'Video has been successfully processed.';
            
            // Hide spinner after showing success message
            setTimeout(() => {
                spinnerOverlay.style.display = 'none';
                if (data.success) {
                    alert('Processing Complete! Output video: ' + data.output_video);
                    // Optionally, you could open the processed video in a new window
                    // window.open(data.output_video, '_blank');
                } else {
                    alert('Processing failed: ' + data.error);
                }
            }, 1500);
        })
        .catch((error) => {
            spinnerText.textContent = 'Processing Failed!';
            spinnerSubtext.textContent = 'An error occurred during processing.';
            
            setTimeout(() => {
                spinnerOverlay.style.display = 'none';
                console.error('Processing error:', error);
                alert('An error occurred during processing. Please try again.');
            }, 2000);
        });
    });
});
