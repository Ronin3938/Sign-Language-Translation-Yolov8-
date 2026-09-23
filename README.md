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

Ensure you have Python installed (v3.8 or higher recommended).

```bash
python --version
