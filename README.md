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


Übungsbeispiel:

Obersatz:
Viktor könnte einen Anspruch auf Zahlung gegen Kurt haben nach §433 2 BGB

Tatbestandmerkmale und Definition:
Voraussetzung dafür ist das Zustandekommen eines wirksamen Kaufvertrags §433 zwischen Viktor und Kurt mit einem einhergehenden Verpflichtungsgeschäft als Teil des Kaufvertags §433 2 BGB. Ein Kaufvertrag setzt zwei übereinstimmende, in Bezug aufeinander abgegebene Willenserklärungen voraus, nämlich Angebot § 145 BGB und Annahme § 147 BGB. Unter einem Verpflichtungsgeschäft versteht man, dass der Käufer verpflichtet ist dem Verkäufer den vereinbarten Kaufpreis zu zahlen und die gekaufte Sache abzunehmen. Ein Angebot ist eine empfangsbedürftige Willenserklärung, die auf den Abschluss eines Vertrags gerichtet ist.
Kurt hat dem Viktor zu verstehen gegeben, dass er sich über das Angebot des Viktors nach §145 das Auto für 5000 Euro zu kaufen, freut.Viktor hat demnach nach §147 seine Annahme getätigt. Es liegen zwei inhaltlich übereinstimmende Willenserklärungen vor, damit ist ein wirksamer Kaufvertrag gemäß § 433 BGB zustande gekommen. Es sind keine Anhaltspunkte dafür ersichtlich, dass der Anspruch erloschen ist, z. B. durch Erfüllung § 362 BGB, Rücktritt § 346 BGB oder Anfechtung § 142 BGB. Der Anspruch müsste auch als Teil des Vetrags durchsetzbar sein. Viktor hat gegen Kurt einen Anspruch auf Zahlung des Kaufpreises in Höhe von 5.000 € gemäß § 433 Abs. 2 BGB.

Gutgläubiger Erwerb

Herr S. könnte Anspruch auf Schadenersatz von Frau M. haben nach §823. Eigentum nach §. Rechtdgutverletzung, Rechtswidrigkeit, 
