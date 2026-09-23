# Real-Time Sign Language Recognition & Translation System

An AI-powered two-way communication platform that translates sign language gestures into text in real time using computer vision and converts written text into sign language representations.

---

## Features

- **Real-Time Hand Gesture Detection:** Detects and classifies 26 alphabet sign language gestures instantly via standard webcam feeds.
- **YOLOv8 Core Model:** High-accuracy gesture recognition powered by an optimized YOLOv8 computer vision model.
- **Text-to-Sign Helper:** Enables users to look up words or letters to view the corresponding sign language outputs.
- **Web Dashboard:** Clean, interactive user interface displaying live video streams, prediction histories, confidence scores, and real-time FPS rates.

---

## System Architecture & Tech Stack

* **Language:** Python
* **Computer Vision & Deep Learning:** OpenCV, YOLOv8 (Ultralytics)
* **Backend Framework:** Flask / Python web services
* **Frontend:** HTML5, CSS3, JavaScript
* **Hardware Requirements:** Standard Webcam, CUDA-compatible GPU (recommended for model training)

---

## Getting Started

### Prerequisites


Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/sign-language-translator.git](https://github.com/your-username/sign-language-translator.git)
cd sign-language-translator
Create and activate a virtual environment (optional but recommended):

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Usage
Run the Application:

Bash
python app.py
Access the Web Interface:
Open your browser and navigate to http://127.0.0.1:5000 (or the local port displayed in your terminal).

Modules:

Real-Time Detection: Grant camera permissions to begin translating live hand gestures into text.

Text-to-Sign Helper: Enter letters or words into the search bar to display sign visuals.

Model Performance
The underlying YOLOv8 detection model demonstrates strong performance metrics across single-hand gesture alphabets:

mAP@50: ~94.9%

Precision: ~92.7%

Inference Speed: Real-time stream support (up to high FPS depending on hardware)

Ensure you have Python installed (v3.8 or higher recommended).

```bash
python --version
