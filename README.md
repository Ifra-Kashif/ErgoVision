# ErgoVision Pro 👁️💻

> **Real-Time Edge AI Ergonomics & Screen Fatigue Monitor**  
> An offline, zero-latency computer vision system engineered to mitigate Computer Vision Syndrome (CVS), forward-head slouching ("Tech Neck"), and dangerous screen proximity during prolonged desk use.

---

## 🌟 Key Features

- **Forward-Head ("Tech Neck") Detection:** Computes the angular vector between ear and shoulder landmarks relative to the vertical axis to flag cervical spine compression.
- **Dynamic Screen Proximity Warning:** Evaluates facial inter-ocular pixel width in real-time to alert users when sitting dangerously close to monitors.
- **State-Machine Blink & Fatigue Engine:** Computes Eye Aspect Ratio (EAR) using 468-point 3D facial mesh geometry; identifies heavy eyelids/micro-sleep and flags dry-eye risks when blinks fall below ergonomic thresholds (< 8 BPM).
- **Lateral Spinal Tracking:** Monitors bilateral shoulder tilt angle to correct unilateral leaning.
- **Privacy-First Edge Processing:** 100% on-device local computation using OpenCV, MediaPipe, and NumPy — no video feed or biometric data ever leaves the local machine.
- **Asynchronous Telemetry Dashboard:** Standalone dark-mode UI built with CustomTkinter and multi-threaded audio alerts (`winsound`) to prevent interface freezes during continuous inference.

---

## 🛠️ Architecture & Tech Stack

- **Language:** Python 3.11
- **Computer Vision & Inference:** MediaPipe (Pose & Face Mesh), OpenCV
- **Math & Spatial Geometry:** NumPy
- **GUI Framework:** CustomTkinter, Pillow (PIL)
- **Audio Signaling:** Asynchronous threading + Windows Native Audio APIs

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/ErgoVision.git
cd ErgoVision