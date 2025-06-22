// Video and canvas overlay logic for drawing oven area rectangle
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

    // Debug function to check canvas state
    function debugCanvas() {
        console.log('Canvas debug info:');
        console.log('- Canvas element:', canvas);
        console.log('- Canvas context:', ctx);
        console.log('- Canvas width:', canvas.width);
        console.log('- Canvas height:', canvas.height);
        console.log('- Canvas style width:', canvas.style.width);
        console.log('- Canvas style height:', canvas.style.height);
        console.log('- Canvas offset width:', canvas.offsetWidth);
        console.log('- Canvas offset height:', canvas.offsetHeight);
        console.log('- Canvas pointer-events:', getComputedStyle(canvas).pointerEvents);
        console.log('- Video loaded:', videoLoaded);
        console.log('- Is drawing:', isDrawing);
        console.log('- Has rect:', hasRect);
    }

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

    // Sync canvas size with video dimensions
    video.addEventListener('loadedmetadata', () => {
        console.log('Video metadata loaded, syncing canvas...');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        video.style.width = '100%';
        video.style.height = '100%';
        videoLoaded = true;
        
        // Ensure canvas is ready for drawing
        canvas.style.pointerEvents = 'auto';
        canvas.classList.remove('drawing');
        
        clearRectAndButton();
        drawOverlay();
        debugCanvas();
        console.log('Video metadata loaded, canvas synced.');
    });

    // Always draw overlay (oven area rectangle) on top of video
    function drawOverlay() {
        if (!ctx || canvas.width === 0 || canvas.height === 0) {
            console.warn('Cannot draw overlay: canvas not ready');
            return;
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (hasRect && rect.width && rect.height) {
            // Draw oven area with special styling
            ctx.strokeStyle = '#ff6b6b'; // Red-orange color for oven area
            ctx.lineWidth = 3;
            ctx.strokeRect(rect.startX, rect.startY, rect.width, rect.height);
            
            // Add oven area label
            ctx.fillStyle = '#ff6b6b';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('OVEN AREA', rect.startX, rect.startY - 10);
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

    // Rectangle drawing logic for oven area
    canvas.addEventListener('mousedown', (e) => {
        console.log('Mouse down event triggered');
        if (!videoLoaded) {
            console.log('mousedown: video not loaded');
            return;
        }
        if (!ctx) {
            console.error('Canvas context not available');
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
        console.log('Started drawing oven area rectangle at:', rect.startX, rect.startY);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) {
            return;
        }
        if (!ctx) {
            console.error('Canvas context not available during mousemove');
            return;
        }
        
        const rectLeft = canvas.getBoundingClientRect();
        const currX = (e.clientX - rectLeft.left) * (canvas.width / canvas.offsetWidth);
        const currY = (e.clientY - rectLeft.top) * (canvas.height / canvas.offsetHeight);
        rect.width = currX - rect.startX;
        rect.height = currY - rect.startY;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'rgba(255,107,107,0.8)'; // Red-orange with transparency
        ctx.lineWidth = 2;
        ctx.setLineDash([6]);
        ctx.strokeRect(rect.startX, rect.startY, rect.width, rect.height);
        ctx.setLineDash([]);
    });

    canvas.addEventListener('mouseup', (e) => {
        console.log('Mouse up event triggered');
        if (!isDrawing) {
            console.log('mouseup: not drawing');
            return;
        }
        isDrawing = false;
        hasRect = Math.abs(rect.width) > 10 && Math.abs(rect.height) > 10;
        canvas.classList.remove('drawing');
        canvas.style.pointerEvents = 'auto'; // Keep pointer events enabled
        drawOverlay();
        clearRectAndButton();
        console.log('Finished drawing oven area rectangle. Has rect:', hasRect, 'Size:', rect.width, 'x', rect.height);
    });

    function clearRectAndButton() {
        // Enable process button only if video loaded and oven area drawn
        const btn = document.getElementById('processBtn');
        if (videoLoaded && hasRect) {
            btn.disabled = false;
            btn.setAttribute('title', '');
        } else {
            btn.disabled = true;
            btn.setAttribute('title', 'Draw an oven area rectangle and upload a video first');
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
                if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
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

    // Handle the oven tracking processing button
    document.getElementById('processBtn').addEventListener('click', () => {
        if (!hasRect || !videoLoaded) {
            alert('Please draw an oven area rectangle on the video first.');
            return;
        }

        // Show loading overlay with oven tracking specific messages
        const spinnerOverlay = document.getElementById('spinnerOverlay');
        const spinnerText = spinnerOverlay.querySelector('.spinner-text');
        const spinnerSubtext = spinnerOverlay.querySelector('.spinner-subtext');
        
        spinnerText.textContent = 'Processing Oven Tracking...';
        spinnerSubtext.textContent = 'Analyzing staff visits to the oven area...';
        spinnerOverlay.style.display = 'flex';
        
        const processData = {
            startX: rect.startX,
            startY: rect.startY,
            width: rect.width,
            height: rect.height,
            filename: uploadedFilename,
            ovenArea: true // Flag to indicate this is oven area tracking
        };

        // Update message after a short delay to show progress
        setTimeout(() => {
            spinnerSubtext.textContent = 'Detecting staff movements and oven interactions...';
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
            spinnerText.textContent = 'Oven Tracking Complete!';
            spinnerSubtext.textContent = 'Staff visits to oven area have been analyzed.';
            
            // Hide spinner after showing success message
            setTimeout(() => {
                spinnerOverlay.style.display = 'none';
                if (data.success) {
                    alert('Oven tracking complete! Processed video: ' + data.output_video + '\n\nStaff visits to the oven area have been tracked and analyzed.');
                    // Optionally, you could open the processed video in a new window
                    // window.open(data.output_video, '_blank');
                } else {
                    alert('Oven tracking failed: ' + data.error);
                }
            }, 1500);
        })
        .catch((error) => {
            spinnerText.textContent = 'Oven Tracking Failed!';
            spinnerSubtext.textContent = 'An error occurred during oven area analysis.';
            
            setTimeout(() => {
                spinnerOverlay.style.display = 'none';
                console.error('Oven tracking error:', error);
                alert('An error occurred during oven tracking. Please try again.');
            }, 2000);
        });
    });

    // Add debug button for testing
    console.log('Canvas drawing system initialized. Use debugCanvas() in console to check state.');
});
