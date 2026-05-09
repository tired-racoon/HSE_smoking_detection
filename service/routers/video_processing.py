import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from collections import deque
from typing import Deque, Dict

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from utils import detect_smoking

router = APIRouter()

job_results: Dict[str, Dict] = {}


async def process_video_smoking_detection(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30

    frame_interval = int(fps * 5)
    frame_count = 0
    sampled_frames = []
    frame_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            sampled_frames.append(frame)
            frame_timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()

    if not sampled_frames:
        return {"verdict": "No", "events": []}

    detection_tasks = []
    for frame in sampled_frames:
        _, buffer = cv2.imencode('.png', frame)
        b64_image = base64.b64encode(buffer).decode('utf-8')
        detection_tasks.append(asyncio.to_thread(detect_smoking, b64_image))

    api_verdicts = await asyncio.gather(*detection_tasks)

    frame_verdicts = []
    events = []
    for i, result in enumerate(api_verdicts):
        if result and "yes" in result.strip().lower():
            frame_verdicts.append("Yes")
            ts = frame_timestamps[i] if i < len(frame_timestamps) else 0
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            events.append({
                "timestamp_seconds": round(ts, 1),
                "timestamp_formatted": f"{minutes:02d}:{seconds:02d}"
            })
        else:
            frame_verdicts.append("No")

    if len(frame_verdicts) < 5:
        verdict = "Yes" if "Yes" in frame_verdicts else "No"
        return {"verdict": verdict, "events": events}

    window_size = 5
    verdicts_window: Deque[str] = deque(maxlen=window_size)
    window_verdicts = []

    for v in frame_verdicts:
        verdicts_window.append(v)
        if len(verdicts_window) == window_size:
            yes_count = verdicts_window.count("Yes")
            window_verdicts.append("Yes" if yes_count / window_size > 0.5 else "No")

    if not window_verdicts:
        return {"verdict": "No", "events": events}

    positive_windows = window_verdicts.count("Yes")
    final_verdict = "Yes" if positive_windows / len(window_verdicts) > 0.5 else "No"
    return {"verdict": final_verdict, "events": events}


async def process_video_and_store_result(job_id: str, video_path: str):
    try:
        result = await process_video_smoking_detection(video_path)
        job_results[job_id] = {
            "status": "completed",
            "verdict": result["verdict"],
            "events": result["events"]
        }
    except Exception as e:
        job_results[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


@router.post("/video/detect-smoking", tags=["Video Processing"], status_code=202)
async def detect_smoking_in_video(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    job_results[job_id] = {"status": "processing"}

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            video_path = tmp.name

        background_tasks.add_task(process_video_and_store_result, job_id, video_path)

        return {
            "message": "Video processing started.",
            "job_id": job_id,
            "status_url": f"/video/detect-smoking/result/{job_id}",
        }
    except Exception as e:
        job_results[job_id] = {"status": "failed", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/detect-smoking/result/{job_id}", tags=["Video Processing"])
async def get_detection_result(job_id: str):
    job = job_results.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job
