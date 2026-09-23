from ultralytics import YOLO
import cv2
import time

# 1. Define the path to your best trained model weights
# IMPORTANT: Use the exact path shown in your previous output.
MODEL_PATH = "D:/CRP Sign/runs/detect/yolov8_sign_language_cli/weights/best.pt"

# 2. Load the custom trained YOLO model
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 3. Initialize webcam capture
# 0 is usually the default camera. If you have multiple cameras, you might need to try 1, 2, etc.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully initialized. Press 'q' to exit.")

# Main loop for real-time inference
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # --- 4. Run Inference on the Frame ---
    # The 'stream=True' argument makes the process more efficient for video streams
    results = model.predict(source=frame, stream=True, verbose=False, conf=0.5)

    # Process and display the results
    for r in results:
        # Get the frame with bounding boxes and labels plotted by YOLOv8
        # This uses the built-in plotting functionality
        annotated_frame = r.plot()
        
        # Display the frame
        cv2.imshow("Sign Language Real-Time Detection (Press 'q' to quit)", annotated_frame)

    # --- 5. Exit condition ---
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Clean up ---
cap.release() # Release the webcam
cv2.destroyAllWindows() # Close all OpenCV windows
print("Program finished.")