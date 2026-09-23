from ultralytics import YOLO
import cv2
import time
import threading
from flask import Flask, Response, render_template, request, url_for
import os
import glob
import numpy as np # Added for robust numpy operations if needed, though not strictly required by your frame code

# --- Configuration ---
# NOTE: Ensure this path is correct for your system
MODEL_PATH = "D:/CRP Sign/runs/detect/yolov8_sign_language_cli/weights/best.pt"
TARGET_FPS = 1.0 # Your requirement: 1 test per 1 second
FRAME_DELAY = 1.0 / TARGET_FPS # Time to wait between inferences

# Image search configuration
IMAGE_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Initialization ---
app = Flask(__name__)
try:
    # 1. Load YOLO model
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None # Set to None if loading fails

# 2. Initialize Video Capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam (cv2.VideoCapture(0)).")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    cap = None # Set to None if webcam fails

# Global state variables for real-time data
last_inference_time = time.time()
current_prediction = "?"
total_detections = 0
history_log = []

# --- Helper Function for Real-time Video Stream (Generator) ---
def generate_frames():
    global last_inference_time, current_prediction, total_detections, history_log
    
    # Check if webcam is available
    if cap is None:
        print("Webcam not available. Cannot stream frames.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip the frame for a mirror effect, common for webcam apps
        frame = cv2.flip(frame, 1)

        # --- Apply the 1 FPS Inference Rate ---
        current_time = time.time()
        
        # Check if enough time has passed since the last inference
        if (current_time - last_inference_time) >= FRAME_DELAY:
            
            # 1. Perform YOLOv8 Inference (Only if model is loaded)
            if model:
                # Use 'conf=0.7' to ensure only high-confidence signs are detected
                # The 'device=0' argument can be added if you want to explicitly use a GPU
                results = model.predict(source=frame, stream=False, verbose=False, conf=0.7) 
                
                # Check results and update global state
                for r in results:
                    # r.boxes.xyxy, r.boxes.conf, r.boxes.cls
                    boxes = r.boxes
                    if len(boxes) > 0 and boxes[0].conf.item() >= 0.7:
                        # Get the most confident detection
                        class_id = int(boxes[0].cls.cpu().numpy())
                        detected_class = model.names[class_id]
                        
                        # Update global state for the web page
                        current_prediction = detected_class
                        total_detections += 1
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        history_log.append(f"[{timestamp}] -> {detected_class}")

            # Update the last inference time
            last_inference_time = current_time
            
        # 2. Draw Annotations for Display
        annotated_frame = frame.copy() 
        
        # Draw the last successful prediction on the frame
        cv2.putText(annotated_frame, f"LAST PREDICTION: {current_prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"TARGET FPS: {TARGET_FPS} | REAL-TIME STREAM", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert frame to JPEG format for web streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame for the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Route 1: Real-Time Detection Dashboard ---

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route that returns the streaming response (Motion JPEG)."""
    # Response requires a generator function for streaming
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data_feed')
def data_feed():
    """Route that returns the current metrics as JSON."""
    # Calculate effective FPS based on the last inference time
    time_since_last = time.time() - last_inference_time
    
    # Avoid division by zero and provide a realistic number
    if total_detections > 0 and time_since_last > 0:
        actual_fps = 1.0 / time_since_last
    else:
        actual_fps = 0.0
        
    # Limit history to the last 7 entries for display
    history_display = '\n'.join(history_log[-7:]) 
    
    return {
        'prediction': current_prediction,
        'total_detections': total_detections,
        'actual_fps': actual_fps,
        'history': history_display
    }

# --- Route 2: Text-to-Image Search Page ---

@app.route('/image_search', methods=['GET', 'POST'])
def image_search():
    """Renders the image search page and handles the keyword search."""
    found_image = None
    search_keyword = None
    
    # Ensure the image folder exists when the app is run
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    
    if request.method == 'POST':
        # 1. Get the keyword from the form
        search_keyword = request.form.get('keyword', '').lower().strip()
        
        if search_keyword:
            # 2. Define the path to search: finds files containing the keyword
            # The search is case-insensitive because search_keyword is lower-cased
            search_pattern = os.path.join(IMAGE_FOLDER, f'*{search_keyword}*.*')
            
            # 3. Search the directory
            matching_files = glob.glob(search_pattern)
            
            if matching_files:
                # 4. Get the first match found and prepare the path for the HTML template
                file_name_with_ext = os.path.basename(matching_files[0])
                
                # Path relative to the 'static' folder for url_for('static', filename=...)
                found_image = f'images/{file_name_with_ext}' 
    
    # Render the template, passing the image path to display it
    return render_template('image_search.html', 
                           found_image=found_image, 
                           search_keyword=search_keyword)

# --- Main Execution ---
if __name__ == '__main__':
    # Run Flask application
    # Note: debug=True is good for development, but for production use a production WSGI server.
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

# --- Cleanup ---
# Ensure webcam resources are released when the app shuts down
# This is typically handled by the application shutting down, but is good practice.
# Note: In a threaded Flask environment, this cleanup is complex to execute reliably on shutdown.
# For simplicity in this dev environment, we rely on Python's garbage collection/OS cleanup.
# if cap:
#     cap.release()