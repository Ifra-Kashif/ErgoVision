# ErgoVision Pro 👁️💻

> **Edge-AI Workstation Ergonomics, Ocular Fatigue & Telemetry Engine**  
> An offline, zero-latency computer vision desktop system designed to mitigate Computer Vision Syndrome (CVS), cervical spine flexion ("Tech Neck"), lateral postural imbalance, and screen proximity fatigue.

[![GitHub Release](https://img.shields.io/github/v/release/Ifra-Kashif/ErgoVision?color=2a9d8f&label=Release)](https://github.com/Ifra-Kashif/ErgoVision/releases)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet)](https://github.com/TomSchimansky/CustomTkinter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📽️ Live Demonstration

<div align="center">
  <img src="assets/demo.gif" alt="ErgoVision Pro Live Demo" width="750" />
</div>

---

## 🌟 Key Technical Features

### 1. Dynamic 3-Second Posture Baseline Calibration
* Implements statistical median sampling over a 3-second initialization window to establish individual baseline coordinates for cervical vertical angle, inter-ocular distance, and lateral shoulder inclination.
* Adapts relative error margins to any desk height, user stature, or webcam angle, preventing false positives.

### 2. Biomechanical Posture & Spinal Geometry
* **Forward-Head ("Tech Neck") Vector:** Tracks cervical angle by computing the arctangent vector between the right ear landmark ($P_{\text{ear}}$) and shoulder joint ($P_{\text{shoulder}}$) relative to the vertical axis:
  $$\theta = \arctan2(|x_{\text{ear}} - x_{\text{shoulder}}|, y_{\text{shoulder}} - y_{\text{ear}})$$
* **Lateral Shoulder Tilt:** Monitors horizontal angular deviation between left and right clavicular points to detect unilateral spinal leaning.

### 3. Ocular Telemetry & Fatigue State Machine
* **Eye Aspect Ratio (EAR):** Computes ocular aperture utilizing 6 facial mesh coordinates per eye:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
* **Dry-Eye Risk Monitoring:** Flags blink deficiency when blink velocity drops below clinical ergonomics thresholds ($< 8\text{ BPM}$).
* **Eyelid Droop & Drowsiness Tracking:** Employs a state transition engine differentiating micro-sleeps and drooping eyelids ($0.20 \le \text{EAR} \le 0.23$) from discrete blinks.

### 4. Screen Proximity Safeguards
* Calculates Euclidean pixel span between facial lateral perimeters (landmarks 33 and 263) against calibrated baseline dimensions, triggering non-blocking alerts when the user sits dangerously close to displays.

### 5. Embedded SQLite Telemetry & Clinical Reporting
* Operates an isolated local database schema (`.data/ergovision.db`) storing time-series monitoring metrics, compliance ratios, and incident counts.
* Aggregates longitudinal performance and calculates composite **Ergonomic Health Scores (0–100)** accompanied by automated clinical observations and corrective guidance.

---

## 🛠️ Architecture & Tech Stack

* **Language:** Python 3.11
* **Inference & Computer Vision:** Google MediaPipe (Pose & Face Mesh pipelines), OpenCV (`cv2`)
* **Spatial & Vector Mathematics:** NumPy
* **GUI Engine:** CustomTkinter (asynchronous UI dispatching, Pillow/PIL rendering)
* **Embedded Database:** SQLite3
* **Audio Alerts:** Multi-threaded non-blocking Windows native audio API (`winsound`)
* **Distribution & Packaging:** PyInstaller standalone binary

---

## 📦 Download & Quickstart

### Option A: Standalone Windows App (No Python Required)
1. Navigate to the [Releases](https://github.com/Ifra-Kashif/ErgoVision/releases) tab.
2. Download `ErgoVisionPro-v1.0.0-Windows.zip`.
3. Extract the archive and double-click `ErgoVisionPro.exe`.

### Option B: Local Source Setup
```bash
# 1. Clone repository
git clone [https://github.com/Ifra-Kashif/ErgoVision.git](https://github.com/Ifra-Kashif/ErgoVision.git)
cd ErgoVision

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Launch telemetry dashboard
python app.py