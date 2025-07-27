# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import cv2
import mediapipe as mp
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Golf AI Backend is running!"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save the uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        temp.flush()
        temp_path = temp.name

    # Initialize pose detection
    mp_pose = mp.solutions.pose.Pose()
    cap = cv2.VideoCapture(temp_path)

    frame_count = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Process every 10th frame for speed
        if frame_count % 10 != 0:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_pose.process(frame_rgb)
        if result.pose_landmarks:
            detected_frames += 1

    cap.release()
    os.remove(temp_path)  # clean up temp file

    if detected_frames == 0:
        feedback = ["No body pose detected. Try a clearer camera angle or better lighting."]
    else:
        feedback = [
            f"Pose detected in {detected_frames} frames.",
            "Great start! Next: implement angle and posture checks."
        ]

    return JSONResponse(content={"feedback": feedback})
