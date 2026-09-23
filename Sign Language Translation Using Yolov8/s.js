// Function to update the metrics panel
function updateMetrics() {
    fetch('/data_feed') // The Flask route for data
        .then(response => response.json())
        .then(data => {
            // Update the big letter prediction
            document.getElementById('current-prediction').textContent = data.prediction;
            
            // Update the detection counter
            document.getElementById('total-detections').textContent = data.total_detections;
            
            // Update the actual FPS (Frame-rate per second)
            document.getElementById('actual-fps').textContent = data.actual_fps.toFixed(1);

            // Update the history log
            document.getElementById('history-log').textContent = data.history;
            
            // Set the status to Active if a detection is made
            const statusElement = document.getElementById('detection-status');
            if (data.prediction !== '?') {
                statusElement.textContent = `DETECTED: ${data.prediction}`;
            } else {
                statusElement.textContent = `Awaiting Hand Sign...`;
            }
        })
        .catch(error => console.error('Error fetching data:', error));
}

// Start fetching data every 500 milliseconds (0.5 seconds)
// This is separate from the 1.0 FPS video inference rate.
setInterval(updateMetrics, 500);

// Set the video source to the Flask stream when the page loads
document.getElementById('video-feed').src = "/video_feed"; 

// Basic control button placeholders
document.getElementById('start-btn').addEventListener('click', () => {
    alert("Starting model inference...");
    // In a full application, this would signal the backend to start the YOLO thread
});

document.getElementById('stop-btn').addEventListener('click', () => {
    alert("Stopping model inference...");
    // In a full application, this would signal the backend to stop the YOLO thread
});