# ESP32-CAM Pose Estimation & Exercise Tracker

Dieses Projekt verwendet einen ESP32-CAM-Videostream und MediaPipe zur Analyse von Körperhaltungen und zum Zählen von Sportübungen wie Push-ups und Squats.

## Features
- Verbindet sich mit einem MJPEG-Stream (z. B. ESP32-CAM)
- Erkennt Posen mithilfe von MediaPipe
- Zählt Push-ups und Squats automatisch (im Moment nur 2 Übungen)
- Modular aufgebaut für einfache Erweiterbarkeit

---
## 🛠 Installation

1. **Repository klonen**:
    ```bash
    git clone https://github.com/lennarddaw/SmartMirror.git
    cd SmartMirror
    ```

2. **Virtuelle Umgebung (optional)**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Projekt im Entwicklungsmodus installieren (optional)**:
    ```bash
    pip install -e .
    ```

---

## Eigenes Model bauen

1. unter /detection_models/

2. neue class ...
def init (immer gleich)
def calculate_angle (immer gleich)
def update()
passe die landmarks an, überlege wie die Bewegung aussieht
32 landmarks: 

## Starten

Passe zuerst die `STREAM_URL` in `main.py` an die IP-Adresse deines ESP32-CAM an:

```python
STREAM_URL = 'http://<individuell>/stream'
