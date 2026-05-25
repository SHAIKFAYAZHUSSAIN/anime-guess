# ANTIGRAVITY 2.0 // Cyberpunk Gesture Anime System

`ANTIGRAVITY 2.0` is a next-generation real-time computer vision system that maps physical hand gestures to iconic anime series, characters, and thematic quote files. 

This is the **Phase 1 MVP**, focusing on extreme tracking stability, high FPS, local asset-caching, and a performance-isolated multi-threaded desktop GUI.

---

## 🌌 System Architecture (Phase 1)
```text
d:/Projects/
│
├── main.py                     # Bootstrap script; launches GUI and hooks detectors
├── gesture_detector.py         # MediaPipe pipeline; hand-landmarks and heuristic classifiers
├── anime_controller.py         # Local data manager; maps gestures to characters and local images
├── ui.py                       # High-performance PySide6 GUI thread and async VideoWorker
├── generate_assets.py          # Procedural neon poster creator for out-of-the-box local setup
├── requirements.txt            # Package list
└── README.md                   # System documentation & user manual
```

---

## ⚡ Core Phase 1 Gestures & Anime Mapping

| Hand Gesture | Mapped Anime | Character | Cyberpunk Visual Theme |
| :--- | :--- | :--- | :--- |
| 👍 **Thumbs Up** | NARUTO | Naruto Uzumaki | 🟠 Hot Orange & Cyber Yellow |
| ✌ **Peace Sign** | ONE PIECE | Monkey D. Luffy | 🟡 Yellow & Crimson Red |
| 👊 **Fist** | ATTACK ON TITAN | Eren Yeager | 🔴 Neon Red & Steel Grey |
| 🖐 **Open Palm** | DEMON SLAYER | Tanjiro Kamado | 🔵 Neon Cyan & Shocking Pink |

---

## ⚙️ Installation & Setup

Follow these simple steps to set up and run the system on your Windows workstation.

### Step 1: Install Dependencies
Open a PowerShell command window and install the required modules from the `requirements.txt` list:
```powershell
pip install -r requirements.txt
```
*(Optionally use `uv pip install -r requirements.txt` if `uv` is installed for extremely fast setup).*

### Step 2: Generate Beautiful Cyber-Poster Assets
To run out-of-the-box without finding custom pictures, run our procedural poster generator script. This will instantly create beautiful, high-resolution cyberpunk neon poster files inside the proper assets folder:
```powershell
python generate_assets.py
```

### Step 3: Run the Application!
Start the system by launching the bootstrap script:
```powershell
python main.py
```

---

## 🕹️ How to Test and Interact

1. **Verify Startup**:
   - The application will initialize the webcam feed within a custom **cyan-bordered viewport** with a real-time overlay showing MediaPipe hand landmarks.
   - The right column acts as the **Holographic Intel Panel**, indicating the system is idle.
2. **Show Gestures**:
   - Hold your hand up facing the webcam.
   - **Thumbs Up** 👍: Fold your index, middle, ring, and pinky fingers down completely while extending your thumb pointing upwards. 
   - **Peace Sign** ✌: Extend only your index and middle fingers in a "V" shape, folding your thumb, ring, and pinky down.
   - **Fist** 👊: Curl all 5 fingers tight into a ball.
   - **Open Palm** 🖐: Spread all 5 fingers wide apart.
3. **Inspect Visual Response**:
   - Upon recognition, the HUD instantly snaps to the character's accent color (e.g. glowing orange for Naruto, neon red for Eren Yeager).
   - The card dynamically loads the procedural poster, displays the character's stats/synopsis, and formats their iconic quote in a high-tech console box.
   - The telemetry bar shows the active gesture name, classification confidence, and a smooth **FPS Counter**.

---

## 🛠️ Diagnostics & Troubleshooting

*   **Camera Feed is Blank/Error**:
    If the viewport says `CAMERA ERROR`, make sure your webcam is plugged in and not in use by another application (Zoom, Teams, etc.). Use the **Camera Source** dropdown in the telemetry footer to switch from `CAM 0` to `CAM 1` or `CAM 2` depending on your active video input device.
*   **Webcam Lagging**:
    To guarantee smooth performance and low latency, `ui.py` isolates camera captures and landmark predictions on an asynchronous background thread. If you notice frame drops, ensure your room has adequate lighting so MediaPipe can isolate hand coordinates quickly.
