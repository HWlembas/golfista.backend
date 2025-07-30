
import os
import json
import tempfile
from typing import Dict, List, Tuple

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Optional: preload lightweight CPU threads for Render/limited CPUs ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Ultralytics YOLOv8 Pose for server-friendly keypoints
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False

from metrics import compute_metrics

app = FastAPI(title="Golfista Backend", version="0.2.0")

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load rules.json once
RULES: List[Dict] = []
try:
    with open("rules.json", "r") as f:
        RULES = json.load(f)
except FileNotFoundError:
    RULES = []

# Lazy-load YOLO model on first request to keep cold-start faster
_yolo_model = None
def get_model():
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        # Smallest pose model; downloads weights on first use (~7-8MB)
        _yolo_model = YOLO("yolov8n-pose.pt")
    return _yolo_model

# COCO keypoint indices to names
KPT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def eval_rule(value, rule: Dict) -> bool:
    op = rule.get("operator")
    thr = rule.get("threshold")
    if value is None:
        return False
    if isinstance(value, float) and (value != value):  # NaN
        return False
    if op == ">":
        return value > thr
    if op == "<":
        return value < thr
    if op == ">=":
        return value >= thr
    if op == "<=":
        return value <= thr
    if op == "==":
        return value == thr
    if op == "between" and isinstance(thr, (list, tuple)) and len(thr) == 2:
        return thr[0] <= value <= thr[1]
    return False

@app.get("/")
def root():
    return {"status": "ok", "yolo_loaded": YOLO_AVAILABLE, "rules_count": len(RULES)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), handedness: str = "right"):
    """
    Accepts a swing video, extracts pose keypoints (YOLOv8-Pose),
    computes metrics, and applies Dewhurst-aligned rules.
    """
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        video_path = tmp.name

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # If YOLO is not available, short-circuit
    if not YOLO_AVAILABLE:
        cap.release()
        os.remove(video_path)
        return JSONResponse(
            content={
                "metrics": {},
                "tips": [],
                "notes": [
                    "YOLOv8-Pose not available on server. Install 'ultralytics' to enable pose extraction."
                ]
            }
        )

    model = get_model()

    # Sample ~10 fps and limit to 8 seconds to keep latency low
    target_fps = 10.0
    step = max(1, int(round(fps / target_fps)))
    max_seconds = 8.0
    max_frames = int(max_seconds * fps)

    frames_keypoints: List[Dict[str, Tuple[float, float]]] = []
    frame_idx = 0
    sampled = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # Run pose model
        results = model(frame, verbose=False)
        if len(results) > 0:
            r = results[0]
            # choose the person with max box confidence if multiple
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints.xy  # shape: [num_persons, 17, 2] in pixels
                confs = r.boxes.conf if r.boxes is not None else None
                person_idx = 0
                if confs is not None and len(confs) > 1:
                    person_idx = int(confs.argmax().item())
                k = kpts[person_idx].cpu().numpy()  # (17,2)

                # Map to expected names
                mapping = {}
                for i, name in enumerate(KPT_NAMES):
                    x, y = float(k[i,0]), float(k[i,1])
                    mapping[name] = (x, y)
                frames_keypoints.append(mapping)

        sampled += 1
        frame_idx += 1
        if frame_idx >= max_frames:
            break

    cap.release()
    os.remove(video_path)

    if len(frames_keypoints) < 6:
        return JSONResponse(
            content={
                "metrics": {},
                "tips": [],
                "notes": ["Not enough keypoints detected. Try a face-on view, full body in frame, and good lighting."]
            }
        )

    # Compute metrics
    metrics = compute_metrics(frames_keypoints, fps=min(target_fps, fps), handedness=handedness)

    # Apply rules
    tips = []
    for rule in RULES:
        value = metrics.get(rule.get("metric"))
        if eval_rule(value, rule):
            tips.append({
                "id": rule.get("id"),
                "metric": rule.get("metric"),
                "value": value,
                "message": rule.get("message"),
                "why": rule.get("why"),
                "source": rule.get("source")
            })

    return JSONResponse(content={
        "metrics": metrics,
        "tips": tips,
        "notes": [
            f"Processed {len(frames_keypoints)} frames (sampled) at ~{min(target_fps, fps):.1f} fps."
        ]
    })
