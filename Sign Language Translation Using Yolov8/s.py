from ultralytics import YOLO
import cv2
import time
import threading
from flask import Flask, Response, render_template

# --- Configuration ---
MODEL_PATH = "D:/CRP Sign/runs/detect/yolov8_sign_language_cli/weights/best.pt"
TARGET_FPS = 1.0 # Your requirement: 1 test per 1 second
FRAME_DELAY = 1.0 / TARGET_FPS # Time to wait between inferences

# --- Initialization ---
app = Flask(__name__)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
last_inference_time = time.time()

current_prediction = "?"
total_detections = 0
history_log = []

# --- Helper Function for Real-time Video Stream (Generator) ---
def generate_frames():
    global last_inference_time, current_prediction, total_detections, history_log
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # --- Apply the 1 FPS Inference Rate ---
        current_time = time.time()
        
        # Check if enough time has passed since the last inference
        if (current_time - last_inference_time) >= FRAME_DELAY:
            
            # 1. Perform YOLOv8 Inference
            # Use 'conf=0.7' to ensure only high-confidence signs are detected
            results = model.predict(source=frame, stream=False, verbose=False, conf=0.7) 
            
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    # Get the most confident detection
                    # 'names' is the class list (A, B, C, etc.)
                    class_id = int(boxes[0].cls.cpu().numpy())
                    detected_class = model.names[class_id]
                    
                    # Update global state for the web page
                    current_prediction = detected_class
                    total_detections += 1
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    history_log.append(f"[{timestamp}] -> {detected_class}")

            # Update the last inference time
            last_inference_time = current_time
            
        # 2. Draw Annotations for Display (Regardless of inference timing)
        # We always plot the frame to send a live video stream, even if we skip the YOLO inference.
        # This keeps the video feed smooth while sampling the inference.
        annotated_frame = frame.copy() 
        # Optionally, you could draw the last successful prediction here if you want:
        cv2.putText(annotated_frame, f"LAST: {current_prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to JPEG format for web streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame for the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Flask Routes ---

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html') # Need to save the HTML above as 'index.html'

@app.route('/video_feed')
def video_feed():
    """Route that returns the streaming response."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data_feed')
def data_feed():
    """Route that returns the current metrics as JSON."""
    # You would typically calculate FPS here, but for simplicity, we use the 1.0 target.
    current_fps = round(1.0 / (time.time() - last_inference_time)) if total_detections > 0 else 0
    
    return {
        'prediction': current_prediction,
        'total_detections': total_detections,
        'actual_fps': current_fps,
        'history': '\n'.join(history_log[-7:]) # Send last 7 predictions
    }

# --- Main Execution ---
if __name__ == '__main__':
    # You will need to rename your main HTML file to 'index.html'
    # and save the CSS to 'style.css' in a 'static' folder.
    # Flask requires you to install it: pip install Flask
    
    # Run Flask application
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)